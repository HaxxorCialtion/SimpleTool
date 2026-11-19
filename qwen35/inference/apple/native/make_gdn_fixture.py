"""Structural FP32 recurrence fixtures; no full model inference or FP16 runs."""
import json
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm.models.gated_delta import normalize_qk,gated_delta_kernel

root=Path('artifacts/native_gdn_fixture');root.mkdir(exist_ok=True)
mx.random.seed(139)
for hk,hv in ((16,32),(16,16),(8,16)):
    folder=root/f'h{hk}_{hv}';folder.mkdir(exist_ok=True)
    prefix=mx.random.normal((1,hv,128,128))*.01
    np.asarray(prefix).tofile(folder/'prefix.bin')
    for batch in (1,2,3,4):
        state=mx.repeat(prefix,batch,axis=0)
        for step in (1,2):
            q,k=normalize_qk(mx.random.normal((batch,1,hk,128)),mx.random.normal((batch,1,hk,128)),
                             inv_scale=128**-.5,eps=1e-6)
            v=mx.random.normal((batch,1,hv,128))*.1
            g=mx.exp(-mx.random.uniform(.01,1.,(batch,1,hv)));beta=mx.sigmoid(mx.random.normal((batch,1,hv)))
            y,state=gated_delta_kernel(q,k,v,g,beta,state)
            for name,array in [('q',q),('k',k),('v',v),('g',g),('beta',beta),('y',y),('state',state)]:
                np.asarray(array).tofile(folder/f'b{batch}_s{step}_{name}.bin')
    (folder/'format.json').write_text(json.dumps({'key_heads':hk,'value_heads':hv,'Dk':128,'Dv':128,
        'batch':[1,2,3,4],'steps':2,'dtype':'float32','prefix_state_batch':1,
        'gold':'installed MLX-LM packed scalar-gate GDN recurrence'},indent=2)+'\n')
