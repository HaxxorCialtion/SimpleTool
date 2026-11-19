"""Actual stored int4 tensors: grouped QMV vs current row-reuse kernel."""
import argparse
import json
import statistics
import time
from pathlib import Path
import mlx.core as mx
from simpletool.loading import load_model
from simpletool.row_reuse_int4 import row_reuse_int4
from simpletool.grouped_qmv import grouped_qmv


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--stream',action='store_true')
    p.add_argument('--repeats',type=int,default=20);args=p.parse_args()
    kernel=grouped_qmv
    if args.stream:
        from simpletool.stream_grouped_qmv import stream_grouped_qmv
        kernel=stream_grouped_qmv
    model,_=load_model(args.model,compact_metadata=True,kernel='custom')
    seen=set();selected=[]
    for name,module in model.named_modules():
        if hasattr(module,'_row_reuse_config'):
            n,k=module.weight.shape[0],module.weight.shape[1]*8
            if k%512==0 and n>=256 and (n,k) not in seen:
                seen.add((n,k));selected.append((name,module))
    records=[];mx.random.seed(173)
    for name,module in selected:
        n,k=module.weight.shape[0],module.weight.shape[1]*8
        for batch in (2,3,4):
            x=mx.random.normal((batch,k));w,s,b=module.weight,module.scales,module.biases
            mx.eval(x,w,s,b)
            variants={'row_reuse':mx.compile(lambda x:row_reuse_int4(x,w,s,b,*module._row_reuse_config))}
            for rows in (2,4,8):
                for threads in (64,128,256):
                    variants[f'r{rows}_t{threads}']=mx.compile(lambda x,r=rows,t=threads:kernel(x,w,s,b,r,t))
            reference=mx.concatenate([mx.quantized_matmul(x[i:i+1],w,s.astype(mx.float32),
                b.astype(mx.float32),group_size=64,bits=4,transpose=True) for i in range(batch)])
            baseline=variants['row_reuse'](x);mx.eval(reference,baseline)
            equal={};errors={}
            for key,fn in variants.items():
                actual=fn(x);mx.eval(actual)
                equal[key]=bool(mx.array_equal(actual,reference).item())
                errors[key]=float(mx.max(mx.abs(actual-reference)).item())
                if key!='row_reuse':assert equal[key],(name,batch,key,errors[key])
            samples={key:[] for key in variants};keys=list(variants)
            for turn in range(args.rounds):
                offset=(turn*3)%len(keys)
                for key in keys[offset:]+keys[:offset]:
                    fn=variants[key];started=time.perf_counter()
                    for _ in range(args.repeats):mx.eval(fn(x))
                    samples[key].append((time.perf_counter()-started)/args.repeats)
            medians={key:statistics.median(v) for key,v in samples.items()};best=min(medians,key=medians.get)
            records.append({'module':name,'N':n,'K':k,'M':batch,'row_reuse_config':module._row_reuse_config,
                'median_s':medians,'samples_s':samples,'bitwise_equal_to_stock_single_qmv':equal,
                'max_abs_error_to_stock_single_qmv':errors,'best':best,
                'compute':'float32','weight_bits':4,'metadata_storage':'lossless float16',
                'rounds':args.rounds,'repeats':args.repeats,'competing_workloads_uncontrolled':True})
            records[-1]['streamed_input']=args.stream
            args.output.write_text(json.dumps(records,indent=2)+'\n')
            print(name,n,k,batch,best,medians['row_reuse']/medians[best],flush=True)


if __name__=='__main__':main()
