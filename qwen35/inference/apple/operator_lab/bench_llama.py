"""Temporary private llama-server baseline; never touches existing services."""
import argparse
import concurrent.futures
import json
import socket
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path


HEADS=("speech","expression","action","command")


def request(url,payload=None):
    if payload is None:
        return json.load(urllib.request.urlopen(url,timeout=10))
    req=urllib.request.Request(url,json.dumps(payload).encode(),{'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req,timeout=180))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=96)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    log=args.output.with_suffix('.server.log').open('w')
    process=subprocess.Popen([str(args.binary),'-m',str(args.model),'-ngl','99','-np','4',
        '-c','4096','-fa','off','--host','127.0.0.1','--port',str(port)],stdout=log,stderr=log)
    try:
        ready=False
        for _ in range(120):
            if process.poll() is not None:raise RuntimeError('private llama-server exited')
            try:
                ready=request(base+'/health').get('status')=='ok'
            except Exception:pass
            if ready:break
            time.sleep(0.25)
        if not ready:raise TimeoutError('private llama-server not ready')
        def one(head):
            payload={'prompt':f'<|im_start|>user\n列出手机设置里的所有选项<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<{head}>',
                     'n_predict':args.tokens,'temperature':0,'cache_prompt':False,
                     'ignore_eos':True,'stream':False}
            start=time.perf_counter()
            response=request(base+'/completion',payload)
            return {'wall_s':time.perf_counter()-start,'timings':response['timings']}
        def four():
            start=time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                rows=list(pool.map(one,HEADS))
            return {'wall_s':time.perf_counter()-start,'heads':rows}
        one('speech'); four()
        samples={'single':[],'four':[]}
        for i in range(3):
            order=[('single',lambda:one('speech')),('four',four)]
            if i%2:order.reverse()
            for key,fn in order:samples[key].append(fn())
        single_tps=statistics.median(r['timings']['predicted_per_second'] for r in samples['single'])
        aggregate=statistics.median(sum(h['timings']['predicted_per_second'] for h in r['heads']) for r in samples['four'])
        report={'model':str(args.model.resolve()),'tokens_per_head':args.tokens,
                'private_server_pid':process.pid,'private_port':port,'server_stopped_on_exit':True,
                'single_decode_tps':single_tps,'aggregate_decode_tps_from_head_timings':aggregate,
                'eta_decode_from_head_timings':aggregate/(4*single_tps),
                'median_four_wall_s':statistics.median(r['wall_s'] for r in samples['four']),
                'median_single_wall_s':statistics.median(r['wall_s'] for r in samples['single']),
                'samples':samples,'cache_prompt':False,'temperature':0,
                'limitations':'GGUF Q4_K_M vs MLX affine int4 are different quantizations; uncontrolled same-machine workloads.'}
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2),flush=True)
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        log.close()


if __name__=='__main__':main()
