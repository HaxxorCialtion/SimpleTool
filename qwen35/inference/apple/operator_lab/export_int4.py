"""Persist unchanged MLX affine-int4 tensors, then verify reload bit for bit."""
import argparse
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import save_model
from simpletool.loading import load_model


def digest(array):
    # Copy one tensor at a time to CPU; never dequantize packed weights.
    import numpy as np
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an existing artifact directory')
    started=time.perf_counter()
    model,tok=load_model(args.model,bits=4,compute='float32')
    args.output.mkdir(parents=True)
    config=json.loads((args.model/'config.json').read_text())
    config['quantization']={'bits':4,'group_size':64,'mode':'affine'}
    config['quantization_config']=config['quantization']
    config['torch_dtype']='float32'
    config.pop('_name_or_path',None)
    # Preserve all architecture metadata, including vision configuration.
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for source in args.model.iterdir():
        if source.is_file() and (source.name.startswith(('tokenizer','special_tokens','added_tokens','chat_template','generation_config','preprocessor','processor'))):
            shutil.copy2(source,args.output/source.name)
    tensors={k:{'shape':list(a.shape),'dtype':str(a.dtype),'bytes':a.nbytes,'sha256':digest(a)}
             for k,a in tree_flatten(model.parameters())}
    save_model(args.output,model)
    del model,tok
    gc.collect();mx.clear_cache()
    reloaded,_=load_model(args.output,bits=4,compute='float32')
    actual=dict(tree_flatten(reloaded.parameters()))
    equal=set(actual)==set(tensors) and all(digest(actual[k])==v['sha256'] for k,v in tensors.items())
    files={}
    for f in args.output.glob('*.safetensors'):
        with f.open('rb') as handle:sha=hashlib.file_digest(handle,'sha256').hexdigest()
        files[f.name]={'bytes':f.stat().st_size,'sha256':sha}
    receipt={'source':str(args.model.resolve()),'artifact':str(args.output.resolve()),
             'quantization':config['quantization'],'floating_compute':'float32',
             'conversion':'MLX RTN affine quantization; no new calibration; no fused parameter renaming',
             'tensor_payload_bytes':sum(v['bytes'] for v in tensors.values()),
             'reload_all_tensors_bitwise_equal':equal,'tensor_count':len(tensors),
             'files':files,'tensors':tensors,'elapsed_s':time.perf_counter()-started}
    (args.output/'int4_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='tensors'},indent=2),flush=True)
    if not equal:raise SystemExit(2)


if __name__=='__main__':main()
