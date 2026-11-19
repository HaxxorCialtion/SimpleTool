"""Export group64 affine RTN INT4 in AWQ layout, NOT activation-aware AWQ.

Source remains immutable. Embedding, output head, vision, tiny GDN gates,
norms and convolution stay FP16. Integer packing is roundtrip verified.
"""
import argparse, hashlib, json, shutil, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', type=Path); ap.add_argument('output', type=Path)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    cfg = json.loads((a.source/'config.json').read_text())
    cfg['quantization_config'] = dict(quant_method='awq', bits=4,
        group_size=64, zero_point=True, version='gemm', lm_head=False,
        modules_to_not_convert=['visual', 'lm_head', 'in_proj_ba', 'in_proj_a', 'in_proj_b', 'mtp'])
    (a.output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    for p in a.source.iterdir():
        if p.is_file() and p.name != 'config.json' and not p.name.startswith('model') and p.suffix != '.safetensors':
            shutil.copy2(p,a.output/p.name)
    targets = ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj',
               'in_proj_qkv','in_proj_z','out_proj')
    inventory=[]; weight_map={}; total_bytes=0; started=time.time()
    idx=json.loads((a.source/'model.safetensors.index.json').read_text());shards=[a.source/n for n in sorted(set(idx['weight_map'].values()))]
    for si, shard in enumerate(shards, 1):
        tensors={}
        with safe_open(shard, framework='pt', device='cpu') as sf:
            for key in sf.keys():
                w=sf.get_tensor(key)
                if '.language_model.' in key and key.endswith('.weight') and key.split('.')[-2] in targets:
                    assert w.ndim==2 and w.shape[1]%64==0 and w.shape[0]%8==0, (key,w.shape)
                    x=w.float().reshape(w.shape[0],-1,64)
                    lo=x.amin(-1); hi=x.amax(-1)
                    scale=((hi-lo)/15).clamp(min=1e-5).half().float()
                    zp=(-lo/scale).round().clamp(0,15).to(torch.int32)
                    q=(x/scale.unsqueeze(-1)).round().to(torch.int32)+zp.unsqueeze(-1)
                    q=q.clamp(0,15).reshape_as(w)
                    order=[0,2,4,6,1,3,5,7]; reverse=[0,4,1,5,2,6,3,7]
                    shifts=torch.arange(0,32,4,dtype=torch.int64)
                    def pack(t):
                        grouped=t.reshape(t.shape[0],-1,8)[:,:,order].to(torch.int64)
                        packed=(grouped<<shifts).sum(-1).to(torch.int32)
                        restored=((packed.long().unsqueeze(-1)>>shifts)&15)[:,:,reverse].reshape_as(t)
                        assert torch.equal(restored,t.long())
                        return packed.contiguous()
                    prefix=key[:-len('weight')]
                    tensors[prefix+'qweight']=pack(q.T.contiguous())
                    tensors[prefix+'qzeros']=pack(zp.T.contiguous())
                    tensors[prefix+'scales']=scale.T.half().contiguous()
                    restored=((q.reshape_as(x).float()-zp.unsqueeze(-1))*scale.unsqueeze(-1))
                    inventory.append(dict(key=key, shape=list(w.shape),
                        rmse=float((restored-x).square().mean().sqrt()),
                        relative_rmse=float((restored-x).square().mean().sqrt()/x.square().mean().sqrt()),
                        packing_roundtrip=True))
                else:
                    tensors[key]=w
        name=f'model-{si:05d}-of-{len(shards):05d}.safetensors'
        save_file(tensors,str(a.output/name),metadata={'format':'pt'})
        for key,value in tensors.items():
            weight_map[key]=name; total_bytes+=value.numel()*value.element_size()
        print(json.dumps(dict(shard=name,quantized_so_far=len(inventory))),flush=True)
        del tensors
    (a.output/'model.safetensors.index.json').write_text(json.dumps(dict(metadata={'total_size':total_bytes},weight_map=weight_map),indent=2)+'\n')
    receipt=dict(method='RTN affine INT4, AWQ serialization; no activation calibration',
        group_size=64,source=str(a.source), output=str(a.output), quantized_modules=inventory,
        total_tensor_bytes=total_bytes,elapsed_seconds=time.time()-started,
        config_sha256=hashlib.sha256((a.source/'config.json').read_bytes()).hexdigest())
    (a.output/'QUANTIZATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='quantized_modules'}),flush=True)

if __name__=='__main__': main()
