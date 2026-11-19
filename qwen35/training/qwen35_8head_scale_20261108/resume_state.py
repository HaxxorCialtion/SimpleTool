"""Restore existing LoRA, tied rows and Adam state without resetting schedules."""
import json
from pathlib import Path

import torch


def restore_model(model, head, directory):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    directory = Path(directory)
    weights = load_file(str(directory / 'adapter_model.safetensors'))
    result = set_peft_model_state_dict(model, weights, adapter_name='default')
    assert not result.unexpected_keys, result.unexpected_keys
    actual = get_peft_model_state_dict(model, adapter_name='default')
    assert actual.keys() == weights.keys()
    for name, value in weights.items():
        assert torch.equal(actual[name].detach().cpu(), value), ('LoRA restoration mismatch', name)
    saved = load_file(str(directory / 'head_tokens.safetensors'))
    assert torch.equal(saved['ids'].long(), head.ids.detach().cpu().long())
    target = saved['rows'].to(device=head.delta.device, dtype=torch.float32)
    with torch.no_grad():
        # Preserve the original frozen semantic-init rows; put the difference
        # in the FP32 delta so both embedding inputs and output logits resume.
        head.delta.copy_(target - head.base.weight[head.ids].float())
    assert torch.equal(head.final_rows(), saved['rows']), 'head rows did not restore exactly'
    return {'verified_lora_tensors': len(weights), 'verified_head_rows': len(saved['ids'])}


def read_state(directory, config, planned, world):
    directory = Path(directory)
    state = torch.load(directory / 'optimizer_state.pt', map_location='cpu', weights_only=True)
    previous = json.loads((directory / 'train_config.json').read_text())
    for key in ['PARALLEL', 'PACKS_PER_GPU_PER_STEP', 'PACKS_PER_FORWARD', 'SEED', 'EPOCHS',
                'LR', 'EMB_LR', 'WARMUP_STEPS', 'LR_SCHEDULE', 'MIN_LR_RATIO',
                'GRAD_CLIP', 'WEIGHT_DECAY', 'LORA_RANK', 'LORA_ALPHA', 'LORA_DROPOUT',
                'HEAD_LOSS_WEIGHTS', 'HEAD_DECAY_START', 'HEAD_DECAY_END',
                'CONTENT_MIN_FACTOR', 'FUNCTION_MIN_FACTOR', 'GRAD_CHECKPOINTING',
                'HEAD_DECAY_ENABLED', 'CE_CHUNK_TOKENS']:
        assert previous[key] == config[key], ('resume config mismatch', key)
    for key in ['DATA_DIR', 'MODEL_DIR']:
        assert Path(previous[key]).resolve() == Path(config[key]).resolve(), key
    old_run = directory.parent
    original = json.loads((old_run / 'run_config.json').read_text())
    assert original['world'] == world
    assert state['planned'] == planned and state['seed'] == config['SEED']
    assert 0 < state['step'] < planned
    steps_per_epoch = planned // config['EPOCHS']
    expected_epoch = (state['step'] - 1) // steps_per_epoch
    expected_offset = ((state['step'] - 1) % steps_per_epoch) * config['PACKS_PER_GPU_PER_STEP']
    assert state['epoch'] == expected_epoch and state['local_offset'] == expected_offset
    return state


def restore_optimizer(optimizer, state):
    optimizer.load_state_dict(state['optimizer'])
    checked = 0
    for group in optimizer.param_groups:
        for param in group['params']:
            saved = optimizer.state[param]
            assert int(saved['step']) == state['step']
            for name in ['exp_avg', 'exp_avg_sq']:
                assert saved[name].shape == param.shape
                assert saved[name].device == param.device and saved[name].dtype == param.dtype
                assert torch.isfinite(saved[name]).all(), name
            checked += 1
    assert checked == len(state['optimizer']['state'])
    return checked
