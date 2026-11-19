import argparse
import json
import statistics
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from simpletool.decode import HEADS, batched_generate, prompt_ids
from simpletool.kernels import install_int4_kernel
from mlx.utils import tree_flatten


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokens", type=int, default=96)
    p.add_argument("--kernel", choices=["mlx", "custom"], default="custom")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compiled", action="store_true")
    p.add_argument("--fused-gdn", action="store_true")
    p.add_argument("--split-projections", action="store_true")
    p.add_argument("--row-reuse", action="store_true")
    p.add_argument("--compute",choices=['float16','float32'],default='float32')
    p.add_argument('--suppress-eos',action='store_true',help='match llama-server ignore_eos sampling policy')
    p.add_argument('--cache-warm',action='store_true',help='reuse exact common-prefix KV in every request')
    args = p.parse_args()
    from simpletool.loading import load_model
    model,tok=load_model(args.model,4,args.compute,args.fused_gdn,args.kernel)
    if args.split_projections:
        if args.fused_gdn or args.kernel!='custom':p.error('split projections requires custom and excludes fused-gdn')
        from simpletool.projection_fusion import fuse_projections
        fuse_projections(model,split_output=True)
    if args.row_reuse:
        if args.split_projections or args.fused_gdn or args.kernel!='custom':p.error('row reuse requires unfused custom projections')
        from simpletool.row_reuse_int4 import install_row_reuse
        install_row_reuse(model)
    from simpletool.compiled import CompiledStep
    ids = prompt_ids(tok, [{"role":"user", "content":"列出手机设置里的所有选项"}])
    banned=tuple(tok.convert_tokens_to_ids(t) for t in ('<|endoftext|>','<|im_end|>')) if args.suppress_eos else ()
    compiled = CompiledStep(model,suppress_tokens=banned) if args.compiled else None
    from simpletool.decode import prefill
    source,source_prefill_s=prefill(model,ids) if args.cache_warm else (None,0.)
    single = lambda: batched_generate(model,tok,ids,args.tokens, heads=HEADS[:1],ignore_stops=True,
                                      compiled_step=compiled,suppress_tokens=banned,prefix_cache=source)
    four = lambda: batched_generate(model,tok,ids,args.tokens,ignore_stops=True,compiled_step=compiled,
                                    suppress_tokens=banned,prefix_cache=source)
    # Warm kernels outside timing; source KV reuse is controlled by --cache-warm.
    warm_single=single(); four()
    tensors=tree_flatten(model.parameters())
    parameter_bytes=sum(t.nbytes for _,t in tensors)
    parameter_bytes_by_dtype={}
    for _,t in tensors:
        key=str(t.dtype);parameter_bytes_by_dtype[key]=parameter_bytes_by_dtype.get(key,0)+t.nbytes
    kv_bytes_per_token=sum(2*layer.self_attn.num_key_value_heads*layer.self_attn.head_dim*4
                           for layer in model.layers if not layer.is_linear)
    samples = {"single":[], "four":[]}
    for round_idx in range(3):
        order = [("single",single),("four",four)]
        if round_idx % 2: order.reverse()
        for name, fn in order:
            result = fn()
            assert all(len(c['token_ids'])==args.tokens and not set(c['token_ids']).intersection(banned)
                       for c in result['channels'].values())
            samples[name].append(result["stats"])
            print(name, round_idx, result["stats"]["total_s"], flush=True)
    t1 = statistics.median(s["decode_s"] for s in samples["single"])
    t4 = statistics.median(s["decode_s"] for s in samples["four"])
    w1 = statistics.median(s["total_s"] for s in samples["single"])
    w4 = statistics.median(s["total_s"] for s in samples["four"])
    report = {"model":str(Path(args.model).resolve()),"tokens_per_head":args.tokens,
              "kernel":args.kernel,"compiled":args.compiled,"fused_gdn":args.fused_gdn,
              "split_projections":args.split_projections,"row_reuse":args.row_reuse,"compute":args.compute,
              "fresh_prefill":not args.cache_warm,"forced_length_ignore_stops":True,
              "exact_prefix_cache_warm":args.cache_warm,"initial_source_prefill_s":source_prefill_s,
              "suppressed_token_ids":list(banned),"prompt_token_ids":ids,
              "resident_parameter_bytes":parameter_bytes,"parameter_bytes_by_dtype":parameter_bytes_by_dtype,
              "gdn_state_bytes_per_branch":warm_single['stats']['batched_gdn_bytes'],
              "full_kv_bytes_per_token_per_branch":kv_bytes_per_token,
              "median_single_decode_s":t1,"median_four_decode_s":t4,
              "single_decode_tps":args.tokens/t1,"aggregate_four_decode_tps":4*args.tokens/t4,
              "eta_decode":t1/t4,"eta_wall":w1/w4,"serial_speedup":4*w1/w4,
              "median_single_total_s":w1,"median_four_total_s":w4,"samples":samples,
              "limitations":"Same-machine competing workloads uncontrolled; no llama.cpp/vLLM/SGLang win claimed."}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "samples"},indent=2),flush=True)


if __name__ == "__main__": main()
