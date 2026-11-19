#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

torch.set_num_threads(4)

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
    assert handle.get_tensor('output_rows').shape == handle.get_tensor('rows').shape == (17, 4096)
    assert handle.metadata()['tied_input_output'] == 'false'
    assert len(handle.get_tensor("ids").unique()) == 17
run_config=json.loads((p/'run_config.json').read_text())
optimizer=torch.load(final/'optimizer_state.pt',map_location='cpu',weights_only=True)
assert optimizer['step']==optimizer['planned']==summary['completed_steps']
assert optimizer['world']==run_config['world']
assert optimizer['optimizer']['state']
for state in optimizer['optimizer']['state'].values():
    assert int(state['step'])==summary['completed_steps']
    assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
for rank in range(run_config['world']):
    rng=torch.load(final/f'rng-rank{rank:03d}.pt',map_location='cpu',weights_only=True)
    assert rng['step']==summary['completed_steps'] and rng['world']==run_config['world']
del optimizer
receipt = {"valid": True, "mode": "LoRA + independent 17 input and 17 output token rows", "checked_tensors": checked,
           "optimizer_finite_and_step_checked":True,"rng_rank_files_checked":run_config['world'],
           "completed_steps": summary["completed_steps"], "validation": summary["validation"]}
(p / "CHECKPOINT_VALIDATION.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
