"""Actual int4 MLP gate/up projections and activation, FP32 compute."""
import argparse,json,statistics,time
from pathlib import Path
import mlx.core as mx
from mlx_lm.models.qwen3_next import swiglu
from simpletool.loading import load_model
from simpletool.fused_swiglu_int4 import fused_swiglu_int4

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();model,_=load_model(args.model,kernel='custom',compact_metadata=True)
    gate,up=model.layers[0].mlp.gate_proj,model.layers[0].mlp.up_proj
    k=gate.weight.shape[1]*8;records=[];mx.random.seed(181)
    for batch in (2,3,4):
        x=mx.random.normal((batch,1,k));mx.eval(x)
        variants={'baseline':mx.compile(lambda x:swiglu(gate(x),up(x)))}
        for rows in (1,2,4,8):
            for threads in (64,128,256):
                variants[f'r{rows}_t{threads}']=mx.compile(lambda x,r=rows,t=threads:fused_swiglu_int4(x,gate,up,r,t))
        ref=variants['baseline'](x);mx.eval(ref);equal={}
        for key,fn in variants.items():
            equal[key]=bool(mx.array_equal(ref,fn(x)).item());assert equal[key],key
        samples={key:[] for key in variants};keys=list(variants)
        for turn in range(5):
            offset=turn*3%len(keys)
            for key in keys[offset:]+keys[:offset]:
                fn=variants[key];started=time.perf_counter()
                for _ in range(30):mx.eval(fn(x))
                samples[key].append((time.perf_counter()-started)/30)
        medians={key:statistics.median(v) for key,v in samples.items()};best=min(medians,key=medians.get)
        records.append({'model':args.model,'M':batch,'N':gate.weight.shape[0],'K':k,'median_s':medians,
            'samples_s':samples,'bitwise_equal':equal,'best':best,'competing_workloads_uncontrolled':True,
            'weight_bits':4,'compute':'float32','model_weight_storage_unchanged':True})
        args.output.write_text(json.dumps(records,indent=2)+'\n');print(batch,best,medians['baseline']/medians[best],flush=True)
if __name__=='__main__':main()
