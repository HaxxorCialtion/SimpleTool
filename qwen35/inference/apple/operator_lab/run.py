import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from simpletool.decode import HEADS, batched_generate, prefill, prompt_ids, serial_generate
from simpletool.kernels import install_int4_kernel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="请把明天上午九点设置成提醒")
    p.add_argument("--system", default=None)
    p.add_argument("--messages", type=Path)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--quantize", type=int, choices=[4, 8], default=4)
    p.add_argument("--kernel", choices=["mlx", "custom"], default="custom")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--compiled", action="store_true")
    p.add_argument("--fused-gdn", action="store_true")
    p.add_argument("--split-projections", action="store_true")
    p.add_argument("--warm-kernels", action="store_true")
    p.add_argument("--row-reuse", action="store_true")
    p.add_argument("--compact-metadata", action="store_true")
    p.add_argument("--cache-limit-mib",type=int)
    p.add_argument("--shared-gdn-seed",action="store_true")
    p.add_argument("--stream-grouped-qmv",action="store_true")
    p.add_argument("--fused-swiglu",action="store_true")
    p.add_argument("--compute", choices=["float16","float32"], default="float32")
    p.add_argument("--output", type=Path)
    p.add_argument("--image", type=Path)
    p.add_argument("--vision-gguf", type=Path)
    p.add_argument("--mmproj", type=Path)
    p.add_argument("--vision-exporter", type=Path, default=Path("artifacts/vision_export"))
    p.add_argument("--vision-cache", type=Path, default=Path("artifacts/vision_cache"))
    args = p.parse_args()
    from simpletool.memory_policy import configure_allocator_cache
    try:allocator_policy=configure_allocator_cache(args.cache_limit_mib)
    except ValueError as error:p.error(str(error))
    if args.compact_metadata and not args.row_reuse:p.error('compact metadata requires --row-reuse')
    if (args.stream_grouped_qmv or args.fused_swiglu) and not args.compact_metadata:p.error('new projection kernels require --compact-metadata')
    index_path = Path(args.model) / "model.safetensors.index.json"
    if index_path.exists():
        files = set(json.loads(index_path.read_text())["weight_map"].values())
        missing = [f for f in files if not (Path(args.model) / f).is_file()]
        if missing:
            raise SystemExit(f"Checkpoint download incomplete: {missing}")
    from simpletool.loading import load_model
    model,tokenizer=load_model(args.model,args.quantize,args.compute,args.fused_gdn,args.kernel,args.compact_metadata)
    if args.split_projections:
        if args.fused_gdn or args.kernel!='custom' or args.quantize!=4:
            p.error('--split-projections requires int4/custom and excludes --fused-gdn')
        from simpletool.projection_fusion import fuse_projections
        fuse_projections(model,split_output=True)
    if args.row_reuse:
        if args.split_projections or args.fused_gdn or args.kernel!='custom' or args.quantize!=4:
            p.error('--row-reuse requires unfused int4/custom projections')
        from simpletool.row_reuse_int4 import install_row_reuse
        if not args.compact_metadata:install_row_reuse(model)
    if args.shared_gdn_seed:
        if args.fused_gdn or args.split_projections or args.compute!='float32':
            p.error('shared GDN seed requires unfused FP32 GDN')
        from simpletool.shared_gdn_seed import install_shared_gdn_seed
        install_shared_gdn_seed(model)
    if args.stream_grouped_qmv:
        from simpletool.stream_grouped_qmv import install_stream_grouped_qmv
        install_stream_grouped_qmv(model)
    if args.fused_swiglu:
        from simpletool.fused_swiglu_int4 import install_fused_swiglu_int4
        install_fused_swiglu_int4(model)
    startup_kernels=None
    if args.warm_kernels:
        from simpletool.kernel_management import warm_int4_kernels
        startup_kernels=warm_int4_kernels(model)
    from simpletool.compiled import CompiledStep
    compiled = CompiledStep(model) if args.compiled else None
    if args.messages:
        messages = json.loads(args.messages.read_text())
    else:
        messages = ([{"role": "system", "content": args.system}] if args.system is not None else [])
        messages.append({"role": "user", "content": args.prompt})
    if args.image:
        if not args.vision_gguf or not args.mmproj:
            p.error("--image requires --vision-gguf and --mmproj")
        from simpletool.vision import image_prefix
        ids, embeddings, positions, next_position, vision_stats = image_prefix(model,
            tokenizer, messages, args.image, args.vision_gguf, args.mmproj,
            args.vision_exporter, args.vision_cache)
        source, pp_seconds = prefill(model, ids, embeddings=embeddings,
                                     positions=positions, next_position=next_position)
        result = batched_generate(model, tokenizer, ids, args.max_tokens, prefix_cache=source,
                                  compiled_step=compiled)
        result["stats"].update(vision_stats)
        result["stats"]["prefill_s"] = pp_seconds
        result["stats"]["total_s"] += pp_seconds + vision_stats["vision_preparation_s"]
    else:
        ids = prompt_ids(tokenizer, messages)
        source = None
        result = batched_generate(model, tokenizer, ids, args.max_tokens, compiled_step=compiled)
    result["model"] = str(Path(args.model).resolve())
    result["kernel"] = args.kernel
    result["fused_gdn"] = args.fused_gdn
    result["split_projections"] = args.split_projections
    result["row_reuse"] = args.row_reuse
    result["compact_metadata"] = getattr(model,'_compact_metadata_receipt',None)
    result["allocator_policy"]=allocator_policy
    result["shared_gdn_seed"]=args.shared_gdn_seed
    result["stream_grouped_qmv"]=args.stream_grouped_qmv
    result["fused_swiglu"]=args.fused_swiglu
    result["kernel_startup"] = startup_kernels
    result["compute"] = args.compute
    result["prompt_token_ids"] = ids
    if args.compare:
        serial = serial_generate(model, tokenizer, ids, args.max_tokens, prefix_cache=source)
        equal = {h: result["channels"][h]["token_ids"] == serial["channels"][h]["token_ids"]
                 for h in HEADS}
        result["serial_reference"] = serial
        result["serial_batch_token_equal"] = equal
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    if args.compare and not all(equal.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
