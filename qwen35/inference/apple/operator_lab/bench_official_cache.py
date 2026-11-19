"""Official unmodified server: cold batch, n_cmpl fork, and slot KV reuse."""
import argparse
import base64
import hashlib
import json
import os
import socket
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

HEADS=('speech','expression','action','command')
COMMIT='6c7a87f7e5e5cd75b8a641c3471f2dee84a6ed17'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=96)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--image',type=Path)
    p.add_argument('--mmproj',type=Path)
    p.add_argument('--flash-attn',choices=['auto','on','off'],default='auto')
    args=p.parse_args()
    if args.image and not args.mmproj:p.error('image requires mmproj')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    question='列出手机设置里的所有选项' if not args.image else '屏幕上现在打开的是什么应用？'
    content=('<__media__>\n' if args.image else '')+question
    prefix=tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,
                                  add_generation_prompt=True,enable_thinking=False)
    image_data=base64.b64encode(args.image.read_bytes()).decode() if args.image else None
    def prompt(text):
        return {'prompt_string':text,'multimodal_data':[image_data]} if image_data else text
    binary=Path('artifacts/llama_official_build/bin/llama-server').resolve()
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    command=[str(binary),'-m',str(args.model.resolve()),'-ngl','99','-np','4','-c','8192',
             '-b','512','-ub','512','-kvu','-cram','0','--no-cache-idle-slots',
             '-fa',args.flash_attn,'--host','127.0.0.1','--port',str(port)]
    if args.image:command+=['--mmproj',str(args.mmproj.resolve()),'--image-max-tokens','1024']
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_METAL_MVEXT_','LLAMA_ARG_'))}
    env['LLAMA_MEDIA_MARKER']='<__media__>'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    log=args.output.with_suffix('.server.log').open('w')
    process=subprocess.Popen(command,stdout=log,stderr=log,env=env)
    defaults={'temperature':0,'ignore_eos':True,'return_tokens':True,'samplers':['temperature'],
              'n_predict':args.tokens,'stream':False}
    samples={key:[] for key in ('single','cold_four','n_cmpl_four','slot_warm_single','slot_warm_four')}
    def http(path,payload=None):
        data=None if payload is None else json.dumps(payload).encode()
        req=urllib.request.Request(base+path,data,{'Content-Type':'application/json'})
        start=time.perf_counter()
        response=json.load(urllib.request.urlopen(req,timeout=240))
        return response,time.perf_counter()-start
    def run(mode):
        prepared=0.;seed=None
        # Unified KV includes idle slots too; remove their prior long prefixes.
        reset,_=http('/completion',{**defaults,'prompt':['<|im_start|>']*4,'n_predict':0,'cache_prompt':False})
        assert len(reset)==4 and len({r['id_slot'] for r in reset})==4
        start=time.perf_counter()
        if mode=='single':
            payload={**defaults,'prompt':prompt(prefix+'<speech>'),'cache_prompt':False,'id_slot':0}
        elif mode=='cold_four':
            payload={**defaults,'prompt':[prompt(prefix+f'<{h}>') for h in HEADS],'cache_prompt':False}
        elif mode=='n_cmpl_four':
            payload={**defaults,'prompt':prompt(prefix+'<speech>'),'n_cmpl':4,'cache_prompt':False}
        elif mode=='slot_warm_single':
            seed,prepared=http('/completion',{**defaults,'prompt':prompt(prefix),'id_slot':0,
                                               'n_predict':0,'cache_prompt':False})
            seed=[seed]
            payload={**defaults,'prompt':prompt(prefix+'<speech>'),'id_slot':0,'cache_prompt':True}
        else:
            # Separate non-child tasks retain their cached prefixes after completion.
            seed,prepared=http('/completion',{**defaults,'prompt':[prompt(prefix)]*4,
                                               'n_predict':0,'cache_prompt':False})
            payload={**defaults,'prompt':[prompt(prefix+f'<{h}>') for h in HEADS],'cache_prompt':True}
        response,wall=http('/completion',payload)
        fresh_total=time.perf_counter()-start
        rows=response if isinstance(response,list) else [response]
        expected=1 if mode in ('single','slot_warm_single') else 4
        assert len(rows)==expected,(mode,len(rows))
        counts=[len(r['tokens']) for r in rows]
        assert all(n==args.tokens for n in counts),(mode,counts)
        if mode.startswith('slot_warm_'):
            expected_prefix=min(r['timings']['prompt_n'] for r in seed)
            assert all(r['timings']['cache_n']>=expected_prefix for r in rows),[r['timings'] for r in rows]
        if mode=='n_cmpl_four':assert len({tuple(r['tokens']) for r in rows})==1,'greedy copies unexpectedly differ'
        stats={'wall_s':wall,'preparation_s':prepared,'fresh_total_including_preparation_s':fresh_total,
               'total_generated_tokens':sum(counts),'token_counts':counts,
               'heads':[{k:r.get(k) for k in ('id_slot','index','timings','tokens','content')} for r in rows],
               'decode_span_s_estimate':max(r['timings']['predicted_ms'] for r in rows)/1000}
        if seed:stats['seed_heads']=[{k:r.get(k) for k in ('id_slot','timings','tokens')} for r in seed]
        return stats
    try:
        ready=False
        for _ in range(240):
            if process.poll() is not None:raise RuntimeError('private official server exited')
            try:ready=http('/health')[0].get('status')=='ok'
            except Exception:pass
            if ready:break
            time.sleep(.25)
        if not ready:raise TimeoutError('official server startup timed out')
        if not args.image:
            server_ids=http('/tokenize',{'content':prefix,'add_special':False})[0]['tokens']
            assert server_ids==tok.encode(prefix,add_special_tokens=False),'HF/GGUF tokenization differs'
        for mode in samples:run(mode)
        modes=list(samples)
        for round_idx in range(3):
            order=modes[round_idx:]+modes[:round_idx]
            for mode in order:
                row=run(mode);samples[mode].append(row)
                print(mode,round_idx,round(row['wall_s'],4),'cached',[h['timings']['cache_n'] for h in row['heads']],flush=True)
        summary={}
        for mode,rows in samples.items():
            summary[mode]={key:statistics.median(r[key] for r in rows) for key in
                 ('wall_s','preparation_s','fresh_total_including_preparation_s','decode_span_s_estimate')}
            summary[mode]['aggregate_wall_tps']=statistics.median(r['total_generated_tokens']/r['wall_s'] for r in rows)
            summary[mode]['aggregate_decode_tps_estimate']=statistics.median(
                (r['total_generated_tokens']-len(r['heads']))/r['decode_span_s_estimate'] for r in rows)
        for mode in summary:
            reference='slot_warm_single' if mode.startswith('slot_warm_') else 'single'
            summary[mode]['single_reference']=reference
            summary[mode]['eta_decode_vs_single_estimate']=summary[reference]['decode_span_s_estimate']/summary[mode]['decode_span_s_estimate']
            summary[mode]['eta_wall_vs_single']=summary[reference]['wall_s']/summary[mode]['wall_s']
        result={'official_commit':COMMIT,'unmodified_upstream_source':True,'binary':str(binary),
                'binary_sha256':hashlib.file_digest(binary.open('rb'),'sha256').hexdigest(),
                'model':str(args.model.resolve()),'tokenizer':str(args.tokenizer.resolve()),
                'image':str(args.image.resolve()) if args.image else None,'prompt_prefix':prefix,
                'tokens_per_head':args.tokens,'samplers':['temperature'],'ignore_eos':True,
                'flash_attn_requested':args.flash_attn,'command':command,'summaries':summary,'samples':samples,
                'idle_slot_cache_reset_before_each_sample':True,'server_pid':process.pid,
                'server_stopped_on_exit':True,'competing_workloads_uncontrolled':True,
                'definitions':{'n_cmpl_four':'One prefill and seq_cp but all four start at speech. Not four distinct channels.',
                    'slot_warm_four':'Four separate prefix prefills, followed by four distinct cached channel prompts. Warm wall excludes setup; fresh_total includes it.',
                    'cold_four':'Four distinct prompts without reuse, submitted in one multi-prompt HTTP call.',
                    'decode_span_s_estimate':'Max per-head predicted_ms; excludes first sampled token and its prompt processing.'}}
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        log.close()


if __name__=='__main__':main()
