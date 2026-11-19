import json
import statistics
from pathlib import Path
import mlx.core as mx
from bench_tuned_kernel import timed
from simpletool.kernels import int4_matmul
from simpletool.matrix_int4 import matrix_int4

def main():
    mx.random.seed(29);rows=[]
    for n,k in [(6144,2048),(2048,6144),(9216,2560),(2560,9216),(65536,2560)]:
        w=mx.random.normal((n,k)).astype(mx.float16)*.02
        p,s,b=mx.quantize(w,bits=4,group_size=64);s=s.astype(mx.float32);b=b.astype(mx.float32)
        x=mx.random.normal((4,k));mx.eval(x,p,s,b);del w
        funcs={'old':lambda:int4_matmul(x,p,s,b), 'matrix_unpadded':lambda:matrix_int4(x,p,s,b,32,64,0)}
        for nt in (32,64):
            for kt in (32,64):
                for pad in (4,8):funcs[f'n{nt}_k{kt}_p{pad}']=lambda nt=nt,kt=kt,pad=pad:matrix_int4(x,p,s,b,nt,kt,pad)
        reference=funcs['old']();mx.eval(reference)
        errors={key:float(mx.max(mx.abs(fn()-reference)).item()) for key,fn in funcs.items()}
        samples={key:[] for key in funcs};keys=list(funcs)
        for i in range(3):
            for key in keys[i*3:]+keys[:i*3]:samples[key].append(timed(funcs[key]))
        med={key:statistics.median(v)*1e3 for key,v in samples.items()}
        row={'N':n,'K':k,'median_ms':med,'max_abs_difference':errors,'samples':samples}
        rows.append(row);print(n,k,med,errors,flush=True)
        Path('reports/matrix_padding_20261008.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
