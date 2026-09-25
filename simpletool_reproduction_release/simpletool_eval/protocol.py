"""Verbatim definitions from now_benchmark/01_inference_multimodel.py; see provenance.json."""
import json


DATASETS_TO_EVAL = [
    # BFCL-v3
    "BFCL_v3_live_simple.json",
    "BFCL_v3_simple.json",
    "BFCL_v3_live_multiple.json",
    "BFCL_v3_multiple.json",
    "BFCL_v3_exec_multiple_dataset.json",
    "BFCL_v3_exec_simple_dataset.json",
    # OpenFunctions V1
    "openfunctions_v1_test.jsonl",
    # Seal-Tools (different tool counts)
    "sealtools_test_1tools.jsonl",
    "sealtools_test_2tools.jsonl",
    "sealtools_test_3tools.jsonl",
    "sealtools_test_4tools.jsonl",
    "sealtools_test_5tools.jsonl",
    "sealtools_test_6tools.jsonl",
    # Seal-Tools (by domain)
    "sealtools_test_in_domain.jsonl",
    "sealtools_test_out_domain.jsonl",
    # ToolAlpaca (converted format)
    "toolalpaca_test_real_converted.jsonl",
    "toolalpaca_test_simulated_converted.jsonl",
    # Mobile Actions
    "mobile_actions.jsonl",
]

SYSTEM_PROMPT_TEMPLATE = """You are a multi-head parallel function calling model. 
## Output Heads

**Head 0 - <content>**: Natural language response
- Format: <content>response text</content>
- Answer what you want to say while you are calling a function

**Head 1 - <function>**: Function names to call
- Format: <function>name</function>
- Name: must match tool defined name

**Head 2-7 - <arg1>、<arg2>、<arg3>、<arg4>、<arg5>、<arg6>**: Function arguments by position
- Format: <argN>value</argN> 
- Strictly fill in according to the parameter order of the tool you intend to call
- Note the special restrictions of parameter definitions for corresponding positions
- If the corresponding tool definition has required parameters, these must be filled in
- Infer the user's actual needs.
- If Unnecessary: <argN><|null|></argN>

**Environment - The information you have.
**History - The tools you have called.
"""

def get_parameter_order(tool):
    if 'function' in tool:
        func = tool['function']
    else:
        func = tool
    return list(func.get('parameters', {}).get('properties', {}).keys())

def build_history_string(history_calls, tools):
    history_strs = []
    tools_map = {t.get('function', t).get('name'): t for t in tools}
    for call in history_calls:
        name = call.get('name')
        args = call.get('args', {})
        param_order = get_parameter_order(tools_map.get(name, {}))
        args_list = []
        for p in param_order:
            if p in args:
                val = args[p]
                args_list.append(f'{p}="{val}"' if isinstance(val, str) else f'{p}={json.dumps(val)}')
        history_strs.append(f"{name}({', '.join(args_list)})")
    return history_strs

def build_multihead_prompt(messages, functions, history_calls=[]):
    context = "<|im_start|>system\n" + SYSTEM_PROMPT_TEMPLATE
    if functions:
        context += "\n## Available Tools:\n\n"
        for tool in functions:
            wrapper = {"type": "function", "function": tool} if 'type' not in tool else tool
            context += json.dumps(wrapper, ensure_ascii=False) + "\n"
    context += "<|im_end|>\n"
    
    history_strs = build_history_string(history_calls, functions)
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    
    context += "<|im_start|>user\n"
    context += "environment: []\n"
    context += f"history: [{', '.join(history_strs)}]\n" if history_strs else "history: []\n"
    context += f"\n{user_msg}<|im_end|>\n"
    context += "<|im_start|>assistant\n"
    return context
