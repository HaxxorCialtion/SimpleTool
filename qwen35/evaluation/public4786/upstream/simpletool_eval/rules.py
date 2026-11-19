"""Verbatim definitions from now_benchmark/051_evaluate_multimodel_final.py; see provenance.json."""
import re
from typing import Dict, List, Any


MAX_PARAMS = 6

BENCHMARK_GROUPS = {
    "BFCL": [
        "BFCL_v3_exec_multiple_dataset",
        "BFCL_v3_exec_simple_dataset",
        "BFCL_v3_live_multiple",
        "BFCL_v3_live_simple",
        "BFCL_v3_multiple",
        "BFCL_v3_simple",
    ],
    "SealTools": {
        "1to6_tools": [
            "sealtools_test_1tools",
            "sealtools_test_2tools",
            "sealtools_test_3tools",
            "sealtools_test_4tools",
            "sealtools_test_5tools",
            "sealtools_test_6tools",
        ],
        "other": [
            "sealtools_test_in_domain",
            "sealtools_test_out_domain",
        ]
    },
    "MobileAct": ["mobile_actions"],
    "OpenFunc": ["openfunctions_v1_test"],
    "ToolAlpaca": ["toolalpaca_test_real_converted", "toolalpaca_test_simulated_converted"],
}

def count_tool_params(tool_def: Dict) -> int:
    if "parameters" in tool_def:
        params = tool_def["parameters"]
        if isinstance(params, dict) and "properties" in params:
            return len(params["properties"])
    if "function" in tool_def and isinstance(tool_def["function"], dict):
        func = tool_def["function"]
        if "parameters" in func:
            params = func["parameters"]
            if isinstance(params, dict) and "properties" in params:
                return len(params["properties"])
    if "properties" in tool_def:
        return len(tool_def["properties"])
    return 0

def has_tool_exceeding_max_params(item_data: Dict, max_params: int = MAX_PARAMS) -> bool:
    tools = None
    for field in ["function", "functions", "tools"]:
        if field in item_data:
            tools = item_data[field]
            break
    if not tools: return False
    if not isinstance(tools, list): tools = [tools]
    for tool in tools:
        if not isinstance(tool, dict): continue
        if count_tool_params(tool) > max_params: return True
    return False

def normalize_string(s: str) -> str:
    if not isinstance(s, str): return str(s)
    return re.sub(r'\s+', ' ', s.lower().strip())

def values_equal(pred_val: Any, gt_val: Any) -> bool:
    if pred_val == gt_val: return True
    if pred_val is None and gt_val == "": return True
    if gt_val is None and pred_val == "": return True
    try:
        if abs(float(pred_val) - float(gt_val)) < 1e-6: return True
    except: pass
    if isinstance(pred_val, str) and isinstance(gt_val, str):
        return normalize_string(pred_val) == normalize_string(gt_val)
    if isinstance(pred_val, list) and isinstance(gt_val, list):
        if len(pred_val) != len(gt_val): return False
        return all(values_equal(p, g) for p, g in zip(pred_val, gt_val))
    return str(pred_val) == str(gt_val)

def check_param_match(pred_val: Any, gt_acceptable_values: List) -> bool:
    if not gt_acceptable_values: return pred_val in [None, ""]
    is_optional = "" in gt_acceptable_values
    if pred_val in [None, ""]: return is_optional
    for val in gt_acceptable_values:
        if val == "": continue
        if values_equal(pred_val, val): return True
    return False

def evaluate_single_prediction(pred_result: List[Dict], gt_list: List[Dict]) -> Dict:
    res = {"func_correct": False, "overall_correct": False}
    if not pred_result:
        if not gt_list: res["func_correct"] = res["overall_correct"] = True
        return res
    if not gt_list: return res
    pred_call = pred_result[0]
    pred_name = pred_call.get("name", "")
    pred_args = pred_call.get("arguments", {})
    for gt_item in gt_list:
        if not gt_item: continue
        gt_name = list(gt_item.keys())[0]
        gt_params = gt_item[gt_name]
        if pred_name != gt_name: continue
        res["func_correct"] = True
        all_params_ok = True
        for p_name, p_acceptable in gt_params.items():
            if not isinstance(p_acceptable, list): p_acceptable = [p_acceptable]
            if not check_param_match(pred_args.get(p_name), p_acceptable):
                all_params_ok = False
                break
        if all_params_ok:
            res["overall_correct"] = True
            break
    return res

def calculate_sealtools_score(model_results: Dict[str, Dict], metric: str = "overall_acc") -> float:
    """
    计算 SealTools 组的分数
    
    逻辑：
    1. 1-6 tools 做 micro average: sum(correct) / sum(total)
    2. 然后与 in_domain, out_domain 做 macro average
    3. 最终: (micro_1to6 + in_domain + out_domain) / 3
    """
    sealtools_config = BENCHMARK_GROUPS["SealTools"]
    
    # Step 1: 计算 1-6 tools 的 micro average
    total_correct_1to6 = 0
    total_samples_1to6 = 0
    
    for ds in sealtools_config["1to6_tools"]:
        if ds in model_results:
            m = model_results[ds]
            total_samples = m.get("total", 0)
            if metric == "overall_acc":
                correct = m.get("overall_correct", 0)
            else:
                correct = m.get("func_correct", 0)
            
            total_correct_1to6 += correct
            total_samples_1to6 += total_samples
    
    micro_avg_1to6 = total_correct_1to6 / total_samples_1to6 if total_samples_1to6 > 0 else 0.0
    
    # Step 2: 获取 in_domain 和 out_domain 的分数
    other_scores = []
    for ds in sealtools_config["other"]:
        if ds in model_results:
            other_scores.append(model_results[ds].get(metric, 0))
    
    # Step 3: macro average of [micro_1to6, in_domain, out_domain]
    all_scores = [micro_avg_1to6] + other_scores
    
    return sum(all_scores) / len(all_scores) if all_scores else 0.0

def calculate_group_score(model_results: Dict[str, Dict], group_name: str, metric: str = "overall_acc") -> float:
    """计算单个 benchmark 组的分数"""
    if group_name == "SealTools":
        return calculate_sealtools_score(model_results, metric)
    
    group_datasets = BENCHMARK_GROUPS.get(group_name, [])
    scores = []
    
    for ds in group_datasets:
        if ds in model_results:
            scores.append(model_results[ds].get(metric, 0))
    
    return sum(scores) / len(scores) if scores else 0.0

def calculate_final_avg(model_results: Dict[str, Dict], metric: str = "overall_acc") -> float:
    """计算最终的 AVG 分数：5个组的 macro average"""
    group_scores = []
    
    for group_name in ["BFCL", "SealTools", "MobileAct", "OpenFunc", "ToolAlpaca"]:
        score = calculate_group_score(model_results, group_name, metric)
        if score > 0:
            group_scores.append(score)
    
    return sum(group_scores) / len(group_scores) if group_scores else 0.0
