"""Verbatim definitions from now_benchmark/040_convert2fc_simpletool.py; see provenance.json."""
import json
import re
from typing import Dict, List, Any, Optional


def get_parameter_order(func_def: Dict) -> List[str]:
    """从函数定义中获取参数顺序"""
    if "function" in func_def:
        func = func_def["function"]
    else:
        func = func_def
    
    params = func.get("parameters", {})
    properties = params.get("properties", {})
    return list(properties.keys())

def find_function_def(func_name: str, functions: List[Dict]) -> Optional[Dict]:
    """根据函数名找到对应的函数定义"""
    for func in functions:
        if "function" in func:
            name = func["function"].get("name", "")
        else:
            name = func.get("name", "")
        
        if name == func_name:
            return func
    return None

def parse_arg_value(arg_value: str) -> Any:
    """解析参数值，处理JSON字符串、空值等"""
    if arg_value is None:
        return None
    
    arg_value = arg_value.strip()
    # 检查是否为空或null标记
    if arg_value == "" or arg_value in ["<|null|>", "|<null>|", "null", "None"]:
        return None
    
    # 尝试解析为JSON (处理 boolean, number, dict, list)
    try:
        parsed = json.loads(arg_value)
        return parsed
    except json.JSONDecodeError:
        pass
    
    return arg_value

def convert_raw_heads_to_fc(raw_heads: Dict[str, str], original_functions: List[Dict]) -> List[Dict]:
    """将 raw_heads 转换为标准的 function calling 格式"""
    result = []
    
    if not raw_heads:
        return result
    
    func_str = raw_heads.get("function", "").strip()
    if not func_str or func_str in ["<|null|>", "None"]:
        return result
    
    # 处理可能的多函数调用 (简单按逗号或换行分割)
    # 注意：如果函数名本身包含特殊字符可能会有问题，但通常符合规范
    func_names = re.split(r'[,;\n]+', func_str)
    func_names = [f.strip() for f in func_names if f.strip() and f not in ["<|null|>", "None"]]
    
    # 获取所有参数值 (arg1 - arg6)
    arg_values = []
    for i in range(1, 7):
        arg_key = f"arg{i}"
        arg_val = raw_heads.get(arg_key, "")
        arg_values.append(arg_val)
    
    # === 情况 A: 单函数调用 ===
    if len(func_names) == 1:
        func_name = func_names[0]
        func_def = find_function_def(func_name, original_functions)
        
        arguments = {}
        if func_def:
            # 有函数定义，严格按照定义顺序填充
            param_order = get_parameter_order(func_def)
            for idx, param_name in enumerate(param_order):
                if idx < len(arg_values):
                    parsed_val = parse_arg_value(arg_values[idx])
                    if parsed_val is not None:
                        arguments[param_name] = parsed_val
        else:
            # 无函数定义（幻觉或检索失败），只能按 arg1, arg2... 填充
            for idx, arg_val in enumerate(arg_values):
                parsed_val = parse_arg_value(arg_val)
                if parsed_val is not None:
                    arguments[f"arg{idx+1}"] = parsed_val
        
        result.append({
            "name": func_name,
            "arguments": arguments
        })

    # === 情况 B: 多函数调用 (Parallel Call) ===
    else:
        # 假设参数是顺序平铺的：FuncA用arg1-2, FuncB用arg3...
        arg_ptr = 0
        for func_name in func_names:
            func_def = find_function_def(func_name, original_functions)
            arguments = {}
            
            if func_def:
                param_order = get_parameter_order(func_def)
                for param_name in param_order:
                    if arg_ptr < len(arg_values):
                        parsed_val = parse_arg_value(arg_values[arg_ptr])
                        if parsed_val is not None:
                            arguments[param_name] = parsed_val
                        arg_ptr += 1
                    else:
                        break
            
            result.append({
                "name": func_name,
                "arguments": arguments
            })
            
    return result
