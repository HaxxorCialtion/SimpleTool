# 冻结4786题：9B / 35B / 122B 新运行时质量评测

结果为冻结本地 acceptable-value matcher 分数，不是官方 BFCL AST / execution 分数。只生成 function 与候选最大参数数量的 arg 头，不生成 content，不执行工具。

|模型|函数名|必填匹配（主指标）|带类型必填|完整本地匹配|全部head闭合|
|---|---:|---:|---:|---:|---:|
|9b-int4-bf16-tp1-full4786|4753/4786 (99.31%)|4320/4786 (90.26%)|4213/4786 (88.03%)|4174/4786 (87.21%)|4786/4786 (100.00%)|
|35b-fp8-bf16-tp1-full4786|4767/4786 (99.60%)|4351/4786 (90.91%)|4243/4786 (88.65%)|4189/4786 (87.53%)|4786/4786 (100.00%)|
|122b-fp8-bf16-tp2-full4786|4764/4786 (99.54%)|4348/4786 (90.85%)|4241/4786 (88.61%)|4195/4786 (87.65%)|4786/4786 (100.00%)|

## 五家族宏平均

按交接协议：BFCL六份等权；Seal先合并1–6候选再与in/out-domain等权；ToolAlpaca两份等权，最后五个家族等权。该宏平均与上方全行micro分数分开。

|模型|函数名macro|必填macro|
|---|---:|---:|
|9b-int4-bf16-tp1-full4786|99.49%|83.59%|
|35b-fp8-bf16-tp1-full4786|99.56%|82.91%|
|122b-fp8-bf16-tp2-full4786|99.63%|82.86%|

## 冻结分母

|数据集|原始输入行|预先过滤行|本轮每模型已评分|
|---|---:|---:|---:|
|BFCL_v3_live_simple|258|14|244|
|BFCL_v3_simple|400|1|399|
|BFCL_v3_live_multiple|1053|169|884|
|BFCL_v3_multiple|200|0|200|
|BFCL_v3_exec_multiple_dataset|50|1|49|
|BFCL_v3_exec_simple_dataset|100|0|100|
|openfunctions_v1_test|109|0|109|
|sealtools_test_1tools|293|1|292|
|sealtools_test_2tools|293|2|291|
|sealtools_test_3tools|293|5|288|
|sealtools_test_4tools|293|6|287|
|sealtools_test_5tools|293|4|289|
|sealtools_test_6tools|293|5|288|
|sealtools_test_in_domain|199|4|195|
|sealtools_test_out_domain|94|0|94|
|toolalpaca_test_real_converted|92|5|87|
|toolalpaca_test_simulated_converted|51|1|50|
|mobile_actions|1283|643|640|

Mobile Actions的原始任务961个，转换输入1283行；过滤321个多调用任务对应643行，最终仅保留640个单调用任务。过滤在推理前冻结，三套模型覆盖完全相同。

## 9b-int4-bf16-tp1-full4786

|数据集|n|函数名|必填匹配|带类型必填|完整本地匹配|head闭合|
|---|---:|---:|---:|---:|---:|---:|
|BFCL_v3_live_simple|244|100.00%|82.38%|81.15%|78.28%|100.00%|
|BFCL_v3_simple|399|100.00%|94.74%|92.73%|93.98%|100.00%|
|BFCL_v3_live_multiple|884|96.83%|87.78%|84.28%|81.56%|100.00%|
|BFCL_v3_multiple|200|99.50%|95.00%|95.00%|94.00%|100.00%|
|BFCL_v3_exec_multiple_dataset|49|97.96%|91.84%|83.67%|85.71%|100.00%|
|BFCL_v3_exec_simple_dataset|100|100.00%|95.00%|85.00%|90.00%|100.00%|
|openfunctions_v1_test|109|100.00%|52.29%|52.29%|49.54%|100.00%|
|sealtools_test_1tools|292|100.00%|91.78%|89.38%|90.41%|100.00%|
|sealtools_test_2tools|291|100.00%|92.44%|90.03%|91.07%|100.00%|
|sealtools_test_3tools|288|100.00%|92.71%|90.28%|91.32%|100.00%|
|sealtools_test_4tools|287|100.00%|92.68%|90.59%|91.29%|100.00%|
|sealtools_test_5tools|289|100.00%|92.04%|89.62%|90.66%|100.00%|
|sealtools_test_6tools|288|99.65%|92.36%|89.93%|90.97%|100.00%|
|sealtools_test_in_domain|195|100.00%|90.26%|87.69%|88.72%|100.00%|
|sealtools_test_out_domain|94|100.00%|96.81%|94.68%|95.74%|100.00%|
|toolalpaca_test_real_converted|87|98.85%|82.76%|82.76%|65.52%|100.00%|
|toolalpaca_test_simulated_converted|50|98.00%|96.00%|90.00%|64.00%|100.00%|
|mobile_actions|640|100.00%|92.03%|92.03%|91.09%|100.00%|

源5%训练文本审计发现 simple_9 / simple_310 重合。保留原4786分母，另报排除后的敏感性：4318/4784 (90.26%)。其他语义或预训练重合没有完成全面审计。

停止原因：`{"closing_marker": 21465}`；替换字符head数：0。

### 错误抽查（固定取前8条主指标错误，不对输出修复）

- `BFCL_v3_live_simple/live_simple_13-3-9`：参考 `get_current_weather`，预测 `get_current_weather`；预测参数 `{"location": "上海，中国", "unit": "celsius"}`；可接受参考值 `{"location": ["Shanghai, China", "\\u4e0a\\u6d77,\\u4e2d\\u56fd"], "unit": ["celsius"]}`；必填匹配 `{"location": false}`。
- `BFCL_v3_live_simple/live_simple_26-6-0`：参考 `uber.ride`，预测 `uber.ride`；预测参数 `{"loc": "123 Đường Đại học, Berkeley, CA 94704", "type": "plus", "time": 10}`；可接受参考值 `{"loc": ["123 Đường Đại học, Berkeley, CA", "123 University Street, Berkeley, CA"], "type": ["plus"], "time": [10]}`；必填匹配 `{"loc": false, "type": true, "time": true}`。
- `BFCL_v3_live_simple/live_simple_29-7-2`：参考 `uber.eat.order`，预测 `uber.eat.order`；预测参数 `{"restaurant": "McDonald's", "items": ["pizza"]}`；可接受参考值 `{"restaurant": ["McDonald's"], "items": [["pizza"]], "quantities": [[1]]}`；必填匹配 `{"restaurant": true, "items": true, "quantities": false}`。
- `BFCL_v3_live_simple/live_simple_34-11-0`：参考 `answer.string`，预测 `answer.string`；预测参数 `{"answer": "**Logistic regression is not present in the text, therefore I cannot answer this question.**"}`；可接受参考值 `{"answer": ["Logistic regression is not present in the text, therefore I cannot answer this question."]}`；必填匹配 `{"answer": false}`。
- `BFCL_v3_live_simple/live_simple_38-15-0`：参考 `get_current_weather`，预测 `get_current_weather`；预测参数 `{"location": "37.8651 N, 119.5383 W", "url": "https://api.open-meteo.com/v1/forecast", "unit": "fahrenheit"}`；可接受参考值 `{"url": ["", "https://api.open-meteo.com/v1/forecast"], "location": ["Mariposa, CA"], "unit": ["fahrenheit"]}`；必填匹配 `{"location": false}`。
- `BFCL_v3_live_simple/live_simple_39-16-0`：参考 `fetch_weather_data`，预测 `fetch_weather_data`；预测参数 `{"latitude": 37.8651, "longitude": -122.3181, "url": "https://api.open-meteo.com/v1/forecast", "units": "metric"}`；可接受参考值 `{"url": ["", "https://api.open-meteo.com/v1/forecast"], "latitude": [37.8651], "longitude": [-119.5383], "units": ["", "metric"], "language": ["", "en"]}`；必填匹配 `{"latitude": true, "longitude": false}`。
- `BFCL_v3_live_simple/live_simple_40-17-0`：参考 `ThinQ_Connect`，预测 `ThinQ_Connect`；预测参数 `{"body": {"body": {"airConJobMode": "AIR_CLEAN", "windStrength": "HIGH", "monitoringEnabled": true}}}`；可接受参考值 `{"body": [{"airConJobMode": ["AIR_CLEAN"], "windStrength": ["HIGH"], "monitoringEnabled": [true], "airCleanOperationMode": ["POWER_ON"], "powerSaveEnabled": ["", false], "coolTargetTemperature": ["", 24], "targetTemperature": ["", 22]}]}`；必填匹配 `{"body": false}`。
- `BFCL_v3_live_simple/live_simple_41-17-1`：参考 `ThinQ_Connect`，预测 `ThinQ_Connect`；预测参数 `{"body": {"body": {"airConJobMode": "AIR_DRY", "windStrength": "MID"}}}`；可接受参考值 `{"body": [{"airConJobMode": ["AIR_DRY"], "windStrength": ["MID", ""], "monitoringEnabled": ["", false], "airCleanOperationMode": ["POWER_OFF", ""], "powerSaveEnabled": ["", false], "coolTargetTemperature": ["", 24], "targetTemperature": ["", 22]}]}`；必填匹配 `{"body": false}`。

## 35b-fp8-bf16-tp1-full4786

|数据集|n|函数名|必填匹配|带类型必填|完整本地匹配|head闭合|
|---|---:|---:|---:|---:|---:|---:|
|BFCL_v3_live_simple|244|100.00%|84.02%|82.79%|80.33%|100.00%|
|BFCL_v3_simple|399|100.00%|94.99%|93.23%|93.73%|100.00%|
|BFCL_v3_live_multiple|884|98.30%|90.05%|86.31%|83.03%|100.00%|
|BFCL_v3_multiple|200|100.00%|95.00%|95.00%|93.00%|100.00%|
|BFCL_v3_exec_multiple_dataset|49|97.96%|91.84%|83.67%|87.76%|100.00%|
|BFCL_v3_exec_simple_dataset|100|100.00%|97.00%|87.00%|92.00%|100.00%|
|openfunctions_v1_test|109|100.00%|46.79%|46.79%|44.04%|100.00%|
|sealtools_test_1tools|292|100.00%|91.78%|89.38%|89.73%|100.00%|
|sealtools_test_2tools|291|100.00%|91.75%|89.35%|90.03%|100.00%|
|sealtools_test_3tools|288|100.00%|92.01%|89.58%|89.93%|100.00%|
|sealtools_test_4tools|287|100.00%|92.33%|90.24%|89.90%|100.00%|
|sealtools_test_5tools|289|100.00%|91.70%|89.27%|89.62%|100.00%|
|sealtools_test_6tools|288|99.65%|92.01%|89.58%|90.62%|100.00%|
|sealtools_test_in_domain|195|100.00%|90.26%|87.69%|88.21%|100.00%|
|sealtools_test_out_domain|94|100.00%|95.74%|93.62%|94.68%|100.00%|
|toolalpaca_test_real_converted|87|98.85%|81.61%|81.61%|63.22%|100.00%|
|toolalpaca_test_simulated_converted|50|98.00%|94.00%|88.00%|62.00%|100.00%|
|mobile_actions|640|100.00%|95.16%|95.16%|95.00%|100.00%|

源5%训练文本审计发现 simple_9 / simple_310 重合。保留原4786分母，另报排除后的敏感性：4349/4784 (90.91%)。其他语义或预训练重合没有完成全面审计。

停止原因：`{"closing_marker": 21465}`；替换字符head数：0。

### 错误抽查（固定取前8条主指标错误，不对输出修复）

- `BFCL_v3_live_simple/live_simple_10-3-6`：参考 `get_current_weather`，预测 `get_current_weather`；预测参数 `{"location": "Yosemite National Park, CA", "unit": "celsius"}`；可接受参考值 `{"location": ["Mariposa, CA"], "unit": ["celsius"]}`；必填匹配 `{"location": false}`。
- `BFCL_v3_live_simple/live_simple_13-3-9`：参考 `get_current_weather`，预测 `get_current_weather`；预测参数 `{"location": "上海, 中国", "unit": "celsius"}`；可接受参考值 `{"location": ["Shanghai, China", "\\u4e0a\\u6d77,\\u4e2d\\u56fd"], "unit": ["celsius"]}`；必填匹配 `{"location": false}`。
- `BFCL_v3_live_simple/live_simple_26-6-0`：参考 `uber.ride`，预测 `uber.ride`；预测参数 `{"loc": "123 Đường Đại học, Berkeley, CA 94704", "type": "plus", "time": 10}`；可接受参考值 `{"loc": ["123 Đường Đại học, Berkeley, CA", "123 University Street, Berkeley, CA"], "type": ["plus"], "time": [10]}`；必填匹配 `{"loc": false, "type": true, "time": true}`。
- `BFCL_v3_live_simple/live_simple_32-9-0`：参考 `answer.string`，预测 `answer.string`；预测参数 `{"answer": "Sure, here is the answer to the question:\\n\\n**Logistic regression is not present in the text, therefore I cannot answer this question.**"}`；可接受参考值 `{"answer": ["Logistic regression is not present in the text, therefore I cannot answer this question."]}`；必填匹配 `{"answer": false}`。
- `BFCL_v3_live_simple/live_simple_33-10-0`：参考 `answer.string`，预测 `answer.string`；预测参数 `{"answer": "Sure, here is the answer to the question:\\n\\n**Logistic regression is not present in the text, therefore I cannot answer this question.**"}`；可接受参考值 `{"answer": ["Logistic regression is not present in the text, therefore I cannot answer this question."]}`；必填匹配 `{"answer": false}`。
- `BFCL_v3_live_simple/live_simple_34-11-0`：参考 `answer.string`，预测 `answer.string`；预测参数 `{"answer": "**Logistic regression is not present in the text, therefore I cannot answer this question.**"}`；可接受参考值 `{"answer": ["Logistic regression is not present in the text, therefore I cannot answer this question."]}`；必填匹配 `{"answer": false}`。
- `BFCL_v3_live_simple/live_simple_38-15-0`：参考 `get_current_weather`，预测 `get_current_weather`；预测参数 `{"location": "Yosemite National Park, CA", "unit": "fahrenheit"}`；可接受参考值 `{"url": ["", "https://api.open-meteo.com/v1/forecast"], "location": ["Mariposa, CA"], "unit": ["fahrenheit"]}`；必填匹配 `{"location": false}`。
- `BFCL_v3_live_simple/live_simple_40-17-0`：参考 `ThinQ_Connect`，预测 `ThinQ_Connect`；预测参数 `{"body": {"airConJobMode": "AIR_CLEAN", "windStrength": "HIGH", "monitoringEnabled": true}}`；可接受参考值 `{"body": [{"airConJobMode": ["AIR_CLEAN"], "windStrength": ["HIGH"], "monitoringEnabled": [true], "airCleanOperationMode": ["POWER_ON"], "powerSaveEnabled": ["", false], "coolTargetTemperature": ["", 24], "targetTemperature": ["", 22]}]}`；必填匹配 `{"body": false}`。

## 122b-fp8-bf16-tp2-full4786

|数据集|n|函数名|必填匹配|带类型必填|完整本地匹配|head闭合|
|---|---:|---:|---:|---:|---:|---:|
|BFCL_v3_live_simple|244|100.00%|86.07%|84.84%|80.74%|100.00%|
|BFCL_v3_simple|399|100.00%|95.24%|93.48%|93.73%|100.00%|
|BFCL_v3_live_multiple|884|97.85%|90.27%|86.65%|83.94%|100.00%|
|BFCL_v3_multiple|200|100.00%|94.00%|94.00%|92.50%|100.00%|
|BFCL_v3_exec_multiple_dataset|49|97.96%|91.84%|83.67%|89.80%|100.00%|
|BFCL_v3_exec_simple_dataset|100|100.00%|96.00%|86.00%|92.00%|100.00%|
|openfunctions_v1_test|109|100.00%|46.79%|46.79%|44.04%|100.00%|
|sealtools_test_1tools|292|100.00%|92.12%|89.73%|90.41%|100.00%|
|sealtools_test_2tools|291|100.00%|92.10%|89.69%|90.38%|100.00%|
|sealtools_test_3tools|288|100.00%|92.01%|89.58%|90.28%|100.00%|
|sealtools_test_4tools|287|100.00%|92.33%|90.24%|90.59%|100.00%|
|sealtools_test_5tools|289|100.00%|92.39%|89.97%|90.66%|100.00%|
|sealtools_test_6tools|288|100.00%|91.32%|88.89%|89.58%|100.00%|
|sealtools_test_in_domain|195|100.00%|89.23%|86.67%|87.69%|100.00%|
|sealtools_test_out_domain|94|100.00%|97.87%|95.74%|96.81%|100.00%|
|toolalpaca_test_real_converted|87|97.70%|81.61%|81.61%|63.22%|100.00%|
|toolalpaca_test_simulated_converted|50|100.00%|96.00%|90.00%|64.00%|100.00%|
|mobile_actions|640|100.00%|93.44%|93.44%|93.28%|100.00%|

源5%训练文本审计发现 simple_9 / simple_310 重合。保留原4786分母，另报排除后的敏感性：4346/4784 (90.84%)。其他语义或预训练重合没有完成全面审计。

停止原因：`{"closing_marker": 21465}`；替换字符head数：0。

### 错误抽查（固定取前8条主指标错误，不对输出修复）

- `BFCL_v3_live_simple/live_simple_26-6-0`：参考 `uber.ride`，预测 `uber.ride`；预测参数 `{"loc": "123 Đường Đại học, Berkeley, CA 94704", "type": "plus", "time": 10}`；可接受参考值 `{"loc": ["123 Đường Đại học, Berkeley, CA", "123 University Street, Berkeley, CA"], "type": ["plus"], "time": [10]}`；必填匹配 `{"loc": false, "type": true, "time": true}`。
- `BFCL_v3_live_simple/live_simple_32-9-0`：参考 `answer.string`，预测 `answer.string`；预测参数 `{"answer": "**Logistic regression is not present in the text, therefore I cannot answer this question.**"}`；可接受参考值 `{"answer": ["Logistic regression is not present in the text, therefore I cannot answer this question."]}`；必填匹配 `{"answer": false}`。
- `BFCL_v3_live_simple/live_simple_33-10-0`：参考 `answer.string`，预测 `answer.string`；预测参数 `{"answer": "Sure, here is the answer to the question:\\n\\n**Logistic regression is not present in the text, therefore I cannot answer this question.**"}`；可接受参考值 `{"answer": ["Logistic regression is not present in the text, therefore I cannot answer this question."]}`；必填匹配 `{"answer": false}`。
- `BFCL_v3_live_simple/live_simple_34-11-0`：参考 `answer.string`，预测 `answer.string`；预测参数 `{"answer": "**Logistic regression is not present in the text, therefore I cannot answer this question.**"}`；可接受参考值 `{"answer": ["Logistic regression is not present in the text, therefore I cannot answer this question."]}`；必填匹配 `{"answer": false}`。
- `BFCL_v3_live_simple/live_simple_35-12-0`：参考 `answer.string`，预测 `answer.string`；预测参数 `{"answer": "Sure, here is the answer to the question:\\n\\n**Logistic regression is not present in the text, therefore I cannot answer this question.**"}`；可接受参考值 `{"answer": ["Logistic regression is not present in the text, therefore I cannot answer this question."]}`；必填匹配 `{"answer": false}`。
- `BFCL_v3_live_simple/live_simple_38-15-0`：参考 `get_current_weather`，预测 `get_current_weather`；预测参数 `{"location": "37.8651, -119.5383", "unit": "fahrenheit"}`；可接受参考值 `{"url": ["", "https://api.open-meteo.com/v1/forecast"], "location": ["Mariposa, CA"], "unit": ["fahrenheit"]}`；必填匹配 `{"location": false}`。
- `BFCL_v3_live_simple/live_simple_40-17-0`：参考 `ThinQ_Connect`，预测 `ThinQ_Connect`；预测参数 `{"body": {"airConJobMode": "AIR_CLEAN", "windStrength": "HIGH", "monitoringEnabled": true}}`；可接受参考值 `{"body": [{"airConJobMode": ["AIR_CLEAN"], "windStrength": ["HIGH"], "monitoringEnabled": [true], "airCleanOperationMode": ["POWER_ON"], "powerSaveEnabled": ["", false], "coolTargetTemperature": ["", 24], "targetTemperature": ["", 22]}]}`；必填匹配 `{"body": false}`。
- `BFCL_v3_live_simple/live_simple_41-17-1`：参考 `ThinQ_Connect`，预测 `ThinQ_Connect`；预测参数 `{"body": {"airConJobMode": "AIR_DRY", "windStrength": "MID"}}`；可接受参考值 `{"body": [{"airConJobMode": ["AIR_DRY"], "windStrength": ["MID", ""], "monitoringEnabled": ["", false], "airCleanOperationMode": ["POWER_OFF", ""], "powerSaveEnabled": ["", false], "coolTargetTemperature": ["", 24], "targetTemperature": ["", 22]}]}`；必填匹配 `{"body": false}`。

## 设置与证据

冻结cases.jsonl（包含prompt与离线参考答案）哈希：`80694c46b01c6e1241010edaa9b2a94db20cbe642877a11655c3a301ebd57b6a`。每头温度0、seed1107、128token上限，原始ChatML无额外空think。参考答案只在客户端评分；类型指标与主指标分列。

9B为5%INT4；35B为5%FP8、基座Qwen3.6-35B-A3B；122B为5%FP8。计算BF16、KV/GDN auto，TP分别1/1/2。与历史FP16计算TP1运行时不同，不声称token parity。

`cloud_receipts/runs/<tag>/`保留run.json、service-config.json、逐head outputs.jsonl、逐题/逐数据集 scores.json。模型服务启动回执另见相邻服务目录；本轮耗时不用于性能比较。

## 评测后控制接口检查（不计入4786题）

三套服务的认证readyz均为HTTP200，未发现OOM或异常退出。对同一个虚拟角色场景，分别关闭/开启content作两次控制请求，所有完成head都复用了完整811-token公共状态，content按开关生成且正常闭合。这是接口/缓存验收，第二次完全相同的动态输入已命中公共缓存，不作为动态场景速度基准。

9B与122B返回合法set_expression调用；35B的intensity输出为<|null|>，导致必填参数缺失，服务显式返回valid=false且不执行工具。35B的自然content正常闭合。该额外案例属于模型/协议输出错误，不能称三套控制样例全部语义正确，也未用它改变冻结4786题分数。

全部逐题、逐数据集重评分与云端汇总一致，见VALIDATION.json。保留122B的2×H200及9B/35B共用的1×H100，旧122B 4×H100已停止。
