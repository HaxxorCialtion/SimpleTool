#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

p = Path(sys.argv[1])
summary = json.loads((p / "training_complete.json").read_text())
assert summary["completed_steps"] == summary["planned_steps"] and summary["skipped_steps"] == 0
final = p / "final"
assert (final / "adapter_config.json").is_file() and (final / "head_tokens.safetensors").is_file()
checked = 0
for file in sorted(final.glob("*.safetensors")):
    with safe_open(str(file), framework="pt") as handle:
        for key in handle.keys():
            value = handle.get_tensor(key)
            assert torch.isfinite(value).all(), (file.name, key)
            checked += 1
with safe_open(str(final / "head_tokens.safetensors"), framework="pt") as handle:
    assert handle.get_tensor("rows").shape[0] == 17
    assert len(handle.get_tensor("ids").unique()) == 17
receipt = {"valid": True, "mode": "LoRA + 17 tied input/output token rows", "checked_tensors": checked,
           "completed_steps": summary["completed_steps"], "validation": summary["validation"]}
(p / "CHECKPOINT_VALIDATION.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
