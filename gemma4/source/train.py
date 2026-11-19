"""One shuffled pass, variable-length isolated packing, DDP and adapter-only artifacts."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from packed_store import Packs as PackStore
from modeling import HEADS, SPECIAL_TOKENS, PackedBias, load_text, attach_adapters, save_adapters, load_adapters


TOKEN_IDS = ()
NULL_ID = -1
HEAD_WEIGHTS = {'content':.01,'function':2.,'arg1':1.1,'arg2':1.2,'arg3':1.3,'arg4':1.4,'arg5':1.4,'arg6':1.4}

def token_weights(pack):
    ids=pack['input_ids']
    result=torch.ones(len(ids),dtype=torch.float32)
    offsets=pack['cu_seqlens'].tolist()
    for name,a,b in zip(pack['sources'],offsets,offsets[1:]):
        head=name.rsplit(':',1)[1];x=ids[a:b]
        w=torch.full((b-a,),HEAD_WEIGHTS[head],dtype=torch.float32)
        w[torch.isin(x,torch.tensor(TOKEN_IDS,dtype=x.dtype))]*=1.5
        if head.startswith('arg'):
            focal={'arg1':1.,'arg2':1.,'arg3':1.5,'arg4':2.,'arg5':2.5,'arg6':3.}[head]
            w[x!=NULL_ID]*=focal
        result[a:b]=w
    return result

def head_factors_at(progress):
    p=min(1.,max(0.,(progress-.5)/.25))
    return {h:low+(1-low)*.5*(1+math.cos(math.pi*p)) for h,low in [('content',.1),('function',.2)]}

def scheduled_weights(batch, factors):
    weights=batch['weights'].clone()
    ends=np.cumsum(batch['lengths']);start=0
    for name,end in zip(batch['sources'],ends):
        factor=factors.get(name.rsplit(':',1)[1],1.)
        weights[(batch['hidden_index']>=start)&(batch['hidden_index']<int(end))]*=factor
        start=int(end)
    return weights


def build_batch(pack, dummy=False):
    ids = pack['input_ids'].long()
    mask = pack['loss_mask'].bool().clone()
    offsets = pack['cu_seqlens'].tolist()
    lengths = tuple(b-a for a,b in zip(offsets, offsets[1:]))
    # Explicitly forbid a next-token target crossing a segment boundary.
    mask[offsets[:-1]] = False
    target_positions = mask.nonzero().flatten()
    weights = token_weights(pack)[target_positions]
    if dummy:
        weights.zero_()
    return {'input_ids':ids[None], 'position_ids':torch.cat([torch.arange(n) for n in lengths])[None],
            'hidden_index':target_positions-1, 'targets':ids[target_positions],
            'weights':weights, 'lengths':lengths, 'sources':pack['sources']}


class Packs(Dataset):
    def __init__(self, packs):
        self.packs = packs
    def __len__(self):
        return len(self.packs)
    def __getitem__(self, index):
        return build_batch(self.packs[max(index,0)], dummy=index < 0)


def identity(value):
    return value


class PackedLM(nn.Module):
    def __init__(self, model, ce_chunk_tokens=128):
        super().__init__()
        self.model = model
        self.chunk = ce_chunk_tokens

    def ce(self, hidden, targets):
        logits = self.model.get_output_embeddings()(hidden).float()
        cap = self.model.config.final_logit_softcapping
        if cap is not None:
            logits = cap * torch.tanh(logits / cap)
        return nn.functional.cross_entropy(logits, targets, reduction='none')

    def forward(self, batch):
        masks = {name:PackedBias(batch['lengths'], window) for name,window in
                 [('full_attention', None), ('sliding_attention', self.model.config.sliding_window)]}
        hidden = self.model.get_decoder()(input_ids=batch['input_ids'], position_ids=batch['position_ids'],
                    attention_mask=masks, use_cache=False).last_hidden_state.flatten(0,1)
        selected = hidden.index_select(0,batch['hidden_index'])
        total = selected.sum() * 0
        # Only supervised positions reach the 262k-vocabulary output layer.
        # Recompute this output/CE chunk in backward, never the decoder unless configured.
        for start in range(0,len(selected),self.chunk):
            end = start+self.chunk
            loss = checkpoint(self.ce, selected[start:end], batch['targets'][start:end], use_reentrant=False)
            total = total + (loss*batch['weights'][start:end]).sum()
        return total


def lr_factor(step, total, warmup, floor):
    warm = max(1,round(total*warmup))
    if step <= warm:
        return step/warm
    progress = (step-warm)/max(1,total-warm)
    return floor+(1-floor)*.5*(1+math.cos(math.pi*min(1,progress)))


def sha(path):
    digest = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(16<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--probe',action='store_true',help='full-model memory check, no training artifacts; discard probe updates')
    ap.add_argument('--allow-world-size-change',action='store_true')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if 'GEMMA_DECODER_GC' in os.environ:
        cfg['gradient_checkpointing'] = os.environ['GEMMA_DECODER_GC']=='1'
    rank,world,local = [int(os.environ.get(k,d)) for k,d in [('RANK','0'),('WORLD_SIZE','1'),('LOCAL_RANK','0')]]
    assert torch.cuda.is_available(), 'GPU training only; use preflight.py for CPU checks'
    torch.cuda.set_device(local)
    device = torch.device('cuda',local)
    if world>1:
        dist.init_process_group('nccl')
    def log(value):
        if rank==0:
            print(json.dumps(value,ensure_ascii=False),flush=True)
    torch.set_num_threads(4)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    data = Path(cfg['data_dir'])
    manifest = json.loads((data/'MANIFEST.json').read_text())
    assert manifest['valid'] and manifest['base_model']==cfg['model_dir']
    assert manifest['epochs']==cfg['epochs']==1
    assert manifest['pack_length']==cfg['max_pack_tokens']<=8192
    packs = PackStore(data)
    if rank==0:
        assert sha(data/'INDEX.json')==manifest['data_sha256']
        packs.verify()
    if world>1:
        dist.barrier()
    packs.order = np.random.default_rng(cfg['seed']).permutation(len(packs))
    tok = AutoTokenizer.from_pretrained(data/'tokenizer',local_files_only=True)
    global TOKEN_IDS,NULL_ID
    TOKEN_IDS=tuple(tok.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS)
    NULL_ID=tok.convert_tokens_to_ids('<|null|>')
    if args.resume:
        assert (args.resume/'COMPLETE.json').is_file(), 'incomplete checkpoint cannot be resumed'
        model = load_adapters(cfg['model_dir'],args.resume,tok,manifest['head_ids'],device)
    else:
        model = attach_adapters(load_text(cfg['model_dir'],device),tok,manifest['head_ids'],cfg['lora_rank'],cfg['lora_alpha'],
                                cfg.get('expert_lora_rank'), cfg.get('expert_lora_alpha'))
    if cfg['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    else:
        model.gradient_checkpointing_disable()
    model.train()
    core = PackedLM(model,cfg['ce_chunk_tokens'])
    trainable = [p for p in core.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable,lr=cfg['lr'],weight_decay=.01,fused=True)
    if args.probe:
        assert world==1 and not args.resume
        # Cover the longest pack and the largest supervised output separately.
        chosen=list(manifest['probe_pack_indices'])
        packs.order=None
        measurements=[]
        for index in chosen:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            batch=build_batch(packs[index])
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            started=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=core(batch)/batch['weights'].sum()
            loss.backward()
            norm=nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            measurements.append({'pack_index':index,'tokens':sum(batch['lengths']),'supervised_tokens':len(batch['targets']),
                'loss':float(loss.detach()),'grad_norm':float(norm),'seconds':time.monotonic()-started,
                'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30})
            log({'event':'full_model_probe_step',**measurements[-1]})
        receipt={'valid':True,'model':cfg['model'],'gradient_checkpointing':cfg['gradient_checkpointing'],
                 'optimizer_allocated':True,'probe_weights_discarded':True,'gpu':torch.cuda.get_device_name(),
                 'measurements':measurements,'time':time.time()}
        receipt_path=Path(cfg['output_dir'])/'MEMORY_PROBE.json'
        receipt_path.parent.mkdir(parents=True,exist_ok=True)
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        return
    # No sampler padding: every real pack is consumed exactly once; only the last
    # distributed step may contain zero-weight dummy forwards for collective parity.
    global_size = cfg['global_packs_per_step']
    assert cfg['micro_packs_per_gpu']==1 and global_size%world==0
    microsteps = global_size//world
    steps = math.ceil(len(packs)/global_size)
    resume_step = 0
    if args.resume:
        state = torch.load(args.resume/'training_state.pt',map_location='cpu',weights_only=False)
        stable=lambda c:{k:v for k,v in c.items() if k not in ('gradient_checkpointing','target_gpu')}
        assert stable(state['config'])==stable(cfg) and state['data_sha256']==manifest['data_sha256']
        changed_world=state['world_size']!=world
        assert not changed_world or args.allow_world_size_change,'world size change requires explicit flag'
        optimizer.load_state_dict(state['optimizer'])
        resume_step = state['step']
        if changed_world:
            # The global pack cursor/batch and cosine schedule are unchanged.
            # Per-rank RNG streams cannot be continued bit-for-bit after repartitioning.
            random.seed(cfg['seed']+rank);np.random.seed(cfg['seed']+rank)
            torch.manual_seed(cfg['seed']+rank);torch.cuda.manual_seed_all(cfg['seed']+rank)
            log({'event':'world_size_changed','old_world_size':state['world_size'],'new_world_size':world,
                 'resume_step':resume_step,'global_packs_per_step':global_size,'rng_reseeded':True})
        else:
            rng = torch.load(args.resume/f'rng_rank{rank}.pt',map_location='cpu',weights_only=False)
            random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state(rng['cuda'])
        del state
    indices = []
    for step in range(resume_step,steps):
        for micro in range(microsteps):
            index=step*global_size+micro*world+rank
            indices.append(index if index<len(packs) else -1)
    kwargs = dict(batch_size=None,sampler=indices,num_workers=cfg['dataloader_workers'],pin_memory=True,collate_fn=identity)
    if cfg['dataloader_workers']:
        kwargs.update(prefetch_factor=cfg['prefetch_factor'],persistent_workers=True,multiprocessing_context='fork')
    loader = iter(DataLoader(Packs(packs),**kwargs))
    if world>1:
        core = DDP(core,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True,bucket_cap_mb=64)
    out = Path(cfg['output_dir'])
    if rank==0:
        out.mkdir(parents=True,exist_ok=True)
        (out/'CONFIG.json').write_text(json.dumps(cfg,indent=2)+'\n')
        (out/'DATA_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    log({'event':'ready','world_size':world,'steps':steps,'packs':len(packs),'trainable_params':sum(p.numel() for p in trainable),
         'decoder_checkpointing':cfg['gradient_checkpointing'],'output_ce_checkpointing':True,'precision':cfg['precision']})

    def save(step, final=False):
        dest = out/('final' if final else 'checkpoint-latest')
        if rank==0:
            dest.mkdir(parents=True,exist_ok=True)
            (dest/'COMPLETE.json').unlink(missing_ok=True)
            save_adapters(model,tok,dest)
            temp=dest/'training_state.tmp'
            torch.save({'step':step,'optimizer':optimizer.state_dict(),'config':cfg,'world_size':world,
                        'data_sha256':manifest['data_sha256']},temp)
            temp.replace(dest/'training_state.pt')
        if world>1:
            dist.barrier()
        torch.save({'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                    'cuda':torch.cuda.get_rng_state()},dest/f'rng_rank{rank}.pt')
        if world>1:
            dist.barrier()
        if rank==0:
            (dest/'COMPLETE.json').write_text(json.dumps({'step':step,'final':final,'world_size':world})+'\n')
        log({'event':'saved','step':step,'path':str(dest)})

    for step in range(resume_step,steps):
        started=time.monotonic()
        factor=lr_factor(step+1,steps,cfg['warmup_fraction'],cfg['min_lr_ratio'])
        for group in optimizer.param_groups:
            group['lr']=cfg['lr']*factor
        head_factors=head_factors_at((step+1)/steps)
        start,end=step*global_size,min(len(packs),(step+1)*global_size)
        denominator=sum(float((token_weights(p)*p['loss_mask']).sum()) for p in packs[start:end])
        optimizer.zero_grad(set_to_none=True)
        loss_total=torch.zeros((),device=device)
        tokens=0
        for micro in range(microsteps):
            batch=next(loader)
            batch={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
            tokens+=sum(batch['lengths'])
            batch['weights']=scheduled_weights(batch,head_factors)
            context=core.no_sync() if world>1 and micro<microsteps-1 else contextlib.nullcontext()
            with context,torch.autocast('cuda',dtype=torch.bfloat16):
                loss=core(batch)
                (loss*(world/denominator)).backward()
            loss_total+=loss.detach()
        norm=nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
        optimizer.step()
        if world>1:
            dist.all_reduce(loss_total)
        torch.cuda.synchronize()
        log({'event':'step','step':step+1,'total_steps':steps,'loss':float(loss_total/denominator),
             'content_factor':head_factors['content'],'function_factor':head_factors['function'],
             'lr':cfg['lr']*factor,'grad_norm':float(norm),'seconds':time.monotonic()-started,
             'peak_GiB':torch.cuda.max_memory_allocated()/2**30})
        request=out/'SAVE_AND_PAUSE'
        requested=torch.tensor(int(request.exists()) if rank==0 else 0,device=device)
        if world>1:
            dist.broadcast(requested,src=0)
        pause=bool(requested.item())
        if step+1<steps and (pause or (step==0 and cfg['save_first_step']) or (step+1)%cfg['save_every_steps']==0):
            save(step+1)
        if pause:
            if rank==0:
                (out/'PAUSED.json').write_text(json.dumps({'step':step+1,'world_size':world,'time':time.time()})+'\n')
                request.unlink(missing_ok=True)
            if world>1:
                dist.destroy_process_group()
            return
    save(steps,final=True)
    if world>1:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
