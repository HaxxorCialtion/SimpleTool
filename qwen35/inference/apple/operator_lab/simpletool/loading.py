import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load


def load_model(path,bits=4,compute='float32',fused_gdn=False,kernel='mlx',compact_metadata=False):
    path=Path(path)
    if bits not in (4,8):raise ValueError('Deployment weights must use int4 or int8')
    if compact_metadata and (bits!=4 or compute!='float32' or fused_gdn or kernel!='custom'):
        raise ValueError('Compact metadata requires unfused int4/custom with FP32 compute')
    config=json.loads((path/'config.json').read_text())
    text_config=config.get('text_config',config)
    if (text_config.get('hidden_size'),text_config.get('num_hidden_layers'))==(2560,32) and bits!=4:
        raise ValueError('SimpleLove 4B deployment requires int4 weights')
    index=path/'model.safetensors.index.json'
    if index.exists():
        files=set(json.loads(index.read_text())['weight_map'].values())
        missing=[f for f in files if not (path/f).is_file()]
        if missing:raise FileNotFoundError(f'Incomplete checkpoint: {missing}')
    model,tok=load(str(path));model.eval()
    if isinstance(model.model.embed_tokens,nn.QuantizedEmbedding) and model.model.embed_tokens.bits!=bits:
        raise ValueError('Stored quantization differs from requested bits; use a matching artifact')
    if bits and not isinstance(model.model.embed_tokens,nn.QuantizedEmbedding):
        nn.quantize(model,bits=bits,group_size=64)
    dtype={'float16':mx.float16,'float32':mx.float32}[compute]
    # Preserve packed values and quantization parameters; change compute precision.
    # A compact artifact can keep its metadata narrow without first expanding
    # the complete scale/bias tensors. Activations and all other floats remain
    # FP32. Standard loading still restores the original artifact exactly.
    metadata_ids={id(module[key]) for _,module in model.named_modules()
                  if compact_metadata and isinstance(module,(nn.QuantizedLinear,nn.QuantizedEmbedding))
                  for key in ('scales','biases')}
    model.apply(lambda a:a.astype(dtype) if mx.issubdtype(a.dtype,mx.floating) and id(a) not in metadata_ids else a)
    if fused_gdn:
        from .fused_gdn import fuse_gdn_projections
        fuse_gdn_projections(model)
    if kernel=='custom':
        from .kernels import install_int4_kernel
        install_int4_kernel(model)
    if compact_metadata:
        from .compact_metadata import install_compact_metadata
        model._compact_metadata_receipt=install_compact_metadata(model)
    mx.eval(model.parameters())
    return model,tok
