# 4786题分项趋势：5%版本

主指标为函数正确且全部必填参数匹配，使用冻结本地匹配器；不跑content。9B INT4、35B FP8、122B FP8，计算BF16。35B基座为Qwen3.6，其余Qwen3.5，精度与基座并非完全控制变量。

|benchmark|n|9B|35B|122B|
|---|---:|---:|---:|---:|
|BFCL_v3_live_simple|244|82.38%|84.02%|86.07%|
|BFCL_v3_simple|399|94.74%|94.99%|95.24%|
|BFCL_v3_live_multiple|884|87.78%|90.05%|90.27%|
|BFCL_v3_multiple|200|95.00%|95.00%|94.00%|
|BFCL_v3_exec_multiple_dataset|49|91.84%|91.84%|91.84%|
|BFCL_v3_exec_simple_dataset|100|95.00%|97.00%|96.00%|
|openfunctions_v1_test|109|52.29%|46.79%|46.79%|
|sealtools_test_1tools|292|91.78%|91.78%|92.12%|
|sealtools_test_2tools|291|92.44%|91.75%|92.10%|
|sealtools_test_3tools|288|92.71%|92.01%|92.01%|
|sealtools_test_4tools|287|92.68%|92.33%|92.33%|
|sealtools_test_5tools|289|92.04%|91.70%|92.39%|
|sealtools_test_6tools|288|92.36%|92.01%|91.32%|
|sealtools_test_in_domain|195|90.26%|90.26%|89.23%|
|sealtools_test_out_domain|94|96.81%|95.74%|97.87%|
|toolalpaca_test_real_converted|87|82.76%|81.61%|81.61%|
|toolalpaca_test_simulated_converted|50|96.00%|94.00%|96.00%|
|mobile_actions|640|92.03%|95.16%|93.44%|

9B→35B主指标增加31题（0.65百分点）；35B→122B减少3题（0.06百分点）。35B正确且122B错误68题；反向65题，不能把净差3题说成只有3题输出不同。三模型主指标共同错误326题，至少一个模型错误577题。

这支持本套任务在当前训练与评分协议下参数扩容收益趋平，不证明训练数据比例已饱和；后者需要同模型尺寸、多训练比例的受控对照。OpenFunctions仍只有46.79–52.29%，函数名100%，错误集中在参数或匹配规则层面，需逐题审计后才能判断模型、标注或匹配器责任。共同错误也不能直接认定参考答案错误。
