"""Verbatim definitions from now_benchmark/051_evaluate_all_checkpoints_3groups.py; see provenance.json."""
from typing import Dict


GROUP_NAMES = ["BFCL", "MobileAct", "Other"]

BENCHMARK_GROUPS = {
    "BFCL": [
        "BFCL_v3_exec_multiple_dataset",
        "BFCL_v3_exec_simple_dataset",
        "BFCL_v3_live_multiple",
        "BFCL_v3_live_simple",
        "BFCL_v3_multiple",
        "BFCL_v3_simple",
    ],
    "MobileAct": ["mobile_actions"],
    "Other": [
        # SealTools 全部
        "sealtools_test_1tools",
        "sealtools_test_2tools",
        "sealtools_test_3tools",
        "sealtools_test_4tools",
        "sealtools_test_5tools",
        "sealtools_test_6tools",
        "sealtools_test_in_domain",
        "sealtools_test_out_domain",
        # OpenFunc
        "openfunctions_v1_test",
        # ToolAlpaca
        "toolalpaca_test_real_converted",
        "toolalpaca_test_simulated_converted",
    ],
}

def calculate_bfcl_score(model_results: Dict[str, Dict], metric: str = "overall_acc") -> float:
    """计算 BFCL 组的分数 - macro average"""
    bfcl_datasets = BENCHMARK_GROUPS["BFCL"]
    scores = []
    
    for ds in bfcl_datasets:
        if ds in model_results:
            scores.append(model_results[ds].get(metric, 0))
    
    return sum(scores) / len(scores) if scores else 0.0

def calculate_mobileact_score(model_results: Dict[str, Dict], metric: str = "overall_acc") -> float:
    """计算 MobileAct 组的分数 - 单个数据集"""
    mobileact_datasets = BENCHMARK_GROUPS["MobileAct"]
    
    for ds in mobileact_datasets:
        if ds in model_results:
            return model_results[ds].get(metric, 0)
    
    return 0.0

def calculate_other_score(model_results: Dict[str, Dict], metric: str = "overall_acc") -> float:
    """
    计算 Other 组的分数 (SealTools全部 + OpenFunc + ToolAlpaca)
    
    逻辑：micro average = 总正确数 / 总样本数
    """
    other_datasets = BENCHMARK_GROUPS["Other"]
    
    total_correct = 0
    total_samples = 0
    
    for ds in other_datasets:
        if ds in model_results:
            m = model_results[ds]
            ds_total = m.get("total", 0)
            if metric == "overall_acc":
                ds_correct = m.get("overall_correct", 0)
            else:  # func_acc
                ds_correct = m.get("func_correct", 0)
            
            total_correct += ds_correct
            total_samples += ds_total
    
    return total_correct / total_samples if total_samples > 0 else 0.0

def calculate_group_score(model_results: Dict[str, Dict], group_name: str, metric: str = "overall_acc") -> float:
    """计算单个 benchmark 组的分数"""
    if group_name == "BFCL":
        return calculate_bfcl_score(model_results, metric)
    elif group_name == "MobileAct":
        return calculate_mobileact_score(model_results, metric)
    elif group_name == "Other":
        return calculate_other_score(model_results, metric)
    else:
        return 0.0

def calculate_final_avg(model_results: Dict[str, Dict], metric: str = "overall_acc") -> float:
    """计算最终的 AVG 分数：3个组的 macro average"""
    group_scores = []
    
    for group_name in GROUP_NAMES:
        score = calculate_group_score(model_results, group_name, metric)
        if score > 0:
            group_scores.append(score)
    
    return sum(group_scores) / len(group_scores) if group_scores else 0.0
