import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_next import swiglu
from simpletool.row_reuse_int4 import row_reuse_int4
from simpletool.loading import load_model

p=argparse.ArgumentParser();p.add_argument('--actual-model',default='artifacts/4b_int4_compact');args=p.parse_args()
root=Path('artifacts/native_swiglu_fixture');root.mkdir(exist_ok=True)
mx.random.seed(191);entries=[]
for n,k in ((79,256),(512,2560),(256,2048)):
    ps=[nn.QuantizedLinear(k,n,bias=False,group_size=64,bits=4) for _ in range(2)]
    # Structural fixture values are deliberately canonicalized to compact
    # metadata, so both storage representations encode identical parameters.
    for m in ps:m.scales=m.scales.astype(mx.float16);m.biases=m.biases.astype(mx.float16)
    entries.append((f'n{n}_k{k}',*ps))
model,_=load_model(args.actual_model,kernel='custom',compact_metadata=True)
entries.append(('actual4b',model.layers[0].mlp.gate_proj,model.layers[0].mlp.up_proj))
for label,gate,up in entries:
    folder=root/label;folder.mkdir(exist_ok=True);n,k=gate.weight.shape[0],gate.weight.shape[1]*8
    for tag,m in [('g',gate),('u',up)]:
        np.asarray(m.weight).tofile(folder/f'{tag}w.bin')
        for key,suffix in [('scales','s'),('biases','b')]:
            array=m[key];np.asarray(array).tofile(folder/f'{tag}{suffix}_half.bin')
            np.asarray(array.astype(mx.float32)).tofile(folder/f'{tag}{suffix}_float.bin')
    for batch in (2,3,4):
        x=mx.random.normal((batch,k));np.asarray(x).tofile(folder/f'x{batch}.bin')
        values=[row_reuse_int4(x,m.weight,m.scales,m.biases,4,128) for m in (gate,up)]
        np.asarray(swiglu(*values)).tofile(folder/f'expected{batch}.bin')
    (folder/'format.json').write_text(json.dumps({'N':n,'K':k,'batch':[2,3,4],
        'actual_model_weights':label=='actual4b','gold':'two separate row-reuse projections followed by MLX swiglu',
        'dtype':'float32','bits':4,'group_size':64},indent=2)+'\n')
    (folder/'sha256.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest()
        for f in folder.glob('*.bin')},indent=2)+'\n')
