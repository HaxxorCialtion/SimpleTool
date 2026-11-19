import argparse
import json
import statistics
from pathlib import Path
from simpletool.decode import batched_generate,prompt_ids
from simpletool.kernels import install_int4_kernel,restore_mlx_kernel
from simpletool.loading import load_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--tokens',type=int,default=96)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    model,tok=load_model(args.model)
    ids=prompt_ids(tok,[{'role':'user','content':'列出手机设置里的所有选项'}])
    modes=[('mlx',restore_mlx_kernel),('custom',install_int4_kernel)]
    for _,set_mode in modes:
        set_mode(model);batched_generate(model,tok,ids,args.tokens,ignore_stops=True)
    samples={key:[] for key,_ in modes}
    for i in range(3):
        for key,set_mode in (modes if i%2==0 else modes[::-1]):
            set_mode(model)
            r=batched_generate(model,tok,ids,args.tokens,ignore_stops=True)
            samples[key].append(r['stats'])
            print(key,i,r['stats']['decode_s'],flush=True)
    medians={key:statistics.median(s['decode_s'] for s in rows) for key,rows in samples.items()}
    result={'compute':'float32','tokens_per_head':args.tokens,'fresh_prefill':True,
            'same_process_alternating':True,'samples':samples,'median_decode_s':medians,
            'custom_vs_mlx_speedup':medians['mlx']/medians['custom'],
            'competing_workloads_uncontrolled':True}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(medians,result['custom_vs_mlx_speedup'])


if __name__=='__main__':main()
