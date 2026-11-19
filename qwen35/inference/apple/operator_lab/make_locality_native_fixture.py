"""Raw FP32 native-Metal inputs and six exact MLX fused-step outputs."""
import hashlib,json
from pathlib import Path
import numpy as np
import mlx.core as mx
from simpletool.fixed_history_gdn import _FUSED_STEP,FUSED_STEP
from simpletool.locality_gdn import HeadLocalityState,ValueLocalityState
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts/locality_native_fixture_20261008';OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(10771);cases=[];sources=['#include <metal_stdlib>\nusing namespace metal;\n']
names=['q','k','v','g','beta','base','history_k','history_g','history_delta','pending']
outputs=['y','delta_out','state_out','history_k_out','history_g_out','history_delta_out']
for hv in (32,16):
 for sb in (1,4):
  for interval,phase,write in [(4,0,False),(4,2,False),(4,3,True),(16,0,False),(16,14,False),(16,15,True)]:
    hk=16;dv=128;b=4;c=interval-1;key=f'h{hv}_sb{sb}_i{interval}_p{phase}'
    def rand(shape):return mx.array(rng.standard_normal(shape).astype(np.float32))
    xs=[rand((b,1,hk,128))*.03,rand((b,1,hk,128))*.03,rand((b,1,hv,dv))*.1,
        mx.exp(-mx.abs(rand((b,1,hv)))),mx.sigmoid(rand((b,1,hv))),rand((sb,hv,dv,128))*.01,
        rand((b,c,hk,128))*.03,mx.exp(-mx.abs(rand((b,c,hv)))),rand((b,c,hv,dv))*.1,mx.array(phase,mx.int32)]
    shapes=[xs[2].shape,(b,hv,dv),(b,hv,dv,128) if write else (1,)]+([(1,)]*3 if write else [xs[6].shape,xs[7].shape,xs[8].shape])
    ys=_FUSED_STEP(inputs=xs,template=[('SB',sb),('C',c),('Hk',hk),('Hv',hv),('Dv',dv),('WRITE',write)],
       grid=(32,dv//8,b*hv),threadgroup=(32,2,1),output_shapes=shapes,output_dtypes=[mx.float32]*6)
    mx.eval(xs,ys);paths=[];golds=[]
    for category,arrays,destination in [('input',xs,paths),('gold',ys,golds)]:
     for index,array in enumerate(arrays):
      path=OUT/f'{key}_{category}{index}.bin';path.write_bytes(np.asarray(array).tobytes());destination.append(str(path))
    functions={};grids={}
    for order,body in [('baseline',FUSED_STEP),('head',HeadLocalityState.sources['fused']),('value',ValueLocalityState.sources['fused'])]:
      fn=f'locality_{key}_{order}';functions[order]=fn;constants=dict(SB=sb,B=b,C=c,Hk=hk,Hv=hv,Dv=dv)
      prefix=''.join(f'#define {k} {v}\n' for k,v in constants.items())+f'#define WRITE {str(write).lower()}\n'
      args=[(f'constant int& {n}' if n=='pending' else f'device const float* {n}')+f' [[buffer({i})]]' for i,n in enumerate(names)]
      args += [f'device float* {n} [[buffer({len(names)+i})]]' for i,n in enumerate(outputs)]
      args += ['uint3 thread_position_in_grid [[thread_position_in_grid]]','uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]']
      sources.append(prefix+'kernel void '+fn+'('+',\n'.join(args)+') {\n'+body+'\n}\n'+''.join(f'#undef {k}\n' for k in (*constants,'WRITE')))
      grids[order]=[32,dv//8*b,hv] if order=='value' and sb==1 else [32,dv//8,b*hv]
    cases.append(dict(key=key,hv=hv,batch=b,stored_batch=sb,interval=interval,phase=phase,write=write,input_files=paths,gold_files=golds,functions=functions,grids=grids))
metal=OUT/'kernels.metal';metal.write_text(''.join(sources))
files=[p for p in OUT.iterdir() if p.suffix in ('.bin','.metal')]
(OUT/'manifest.json').write_text(json.dumps(dict(cases=cases,metal_source=str(metal),sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
print(len(cases),'raw fixtures / 72 native pipelines',flush=True)
