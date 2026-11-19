"""Micro A/B on the actual stored int4 tied embedding; no dense expansion."""
import argparse
import json
import statistics
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from simpletool.row_reuse_int4 import install_row_reuse
from simpletool.compact_metadata import install_compact_metadata
from simpletool.greedy_head import greedy_int4


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--repeats',type=int,default=30);args=p.parse_args()
    loaded=mx.load(str(args.model/'model.safetensors'))
    name=next(k[:-7] for k in loaded if k.endswith('embed_tokens.weight'))
    w,s,b=(loaded[name+'.'+key] for key in ('weight','scales','biases'))
    del loaded
    root=nn.Module();root.head=nn.QuantizedEmbedding(w.shape[0],w.shape[1]*8,group_size=64,bits=4)
    root.head.weight=w;root.head.scales=s;root.head.biases=b
    install_row_reuse(root);install_compact_metadata(root);head=root.head
    mx.eval(head.parameters());mx.clear_cache();mx.random.seed(131)
    results=[]
    for m in (4,3,2,1):
        x=mx.random.normal((m,w.shape[1]*8));suppressed=(0,1)
        def standard(x):
            logits=head.as_linear(x);logits[:,list(suppressed)]=-mx.inf
            return mx.argmax(logits,axis=-1)
        def fused(x):return greedy_int4(head,x,suppressed)
        variants={'standard':standard,'fused':fused,'compiled_standard':mx.compile(standard),'compiled_fused':mx.compile(fused)}
        expected=standard(x);equal={}
        for key,fn in variants.items():
            value=fn(x);mx.eval(value);equal[key]=mx.array_equal(value,expected).item()
        assert all(equal.values()),equal
        samples={key:[] for key in variants};keys=list(variants)
        for i in range(args.rounds):
            for key in keys[i%4:]+keys[:i%4]:
                fn=variants[key]
                for _ in range(3):mx.eval(fn(x))
                start=time.perf_counter()
                for _ in range(args.repeats):mx.eval(fn(x))
                samples[key].append((time.perf_counter()-start)/args.repeats)
        medians={key:statistics.median(v)*1000 for key,v in samples.items()}
        rows,threads=head._row_reuse_config;tiles=(w.shape[0]+rows*(threads//32)-1)//(rows*(threads//32))
        results.append({'M':m,'N':w.shape[0],'K':w.shape[1]*8,'equal':equal,'median_ms':medians,
            'samples_s':samples,'full_logits_bytes':m*w.shape[0]*4,
            'fused_partial_bytes':m*tiles*8 if m>1 else m*w.shape[0]*4,'row_config':list(head._row_reuse_config),
            'compute':'float32','weights':'stored affine int4 group64','metadata':'lossless float16 storage',
            'competing_workloads_uncontrolled':True})
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        print(m,medians,'equal',equal,flush=True)


if __name__=='__main__':main()
