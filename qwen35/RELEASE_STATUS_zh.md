# SimpleTool 首版开源交付核对

核对日期：2026-11-19。本版新增在 `qwen35/`，保留原论文及 VLM 仓库的边界。

## 已经到位

- 七个尺寸，11 份公开权重：0.8B／2B／4B／9B 各 BF16 与 INT4；35B-A3B／122B-A10B／397B-A17B 为 FP8。35B 基座实际为 Qwen3.6，其他为 Qwen3.5。
- HF 与 ModelScope 匿名可见，191 个权重分片的大小和 SHA256 元数据逐项相符，索引指向的分片都存在。见 [模型目录](models/README.md)、[模型版本](models/registry.json)。此次没有再下载数百 GiB 权重，也没有把元数据核对说成一次新推理验收。
- 云端真实训练源码已取回：小稠密模型、9B、MoE 三组；包括训练、packing、空 think 处理、loss mask、LoRA 与 special-token 行、合并与导出，以及精确抽样／分组／pack 哈希清单。
- CUDA 核心：零输出公共 prefill、精确末端 KV／GDN／conv 状态发布、跨 head 分叉、独立可写递归状态、TP worker、有界图缓存、逐头结束及可选 content。397B FP8 使用 BF16 计算以避开已诊断的 FP16 中间值溢出。
- Apple 核心：完整 Swift 引擎和实际 MLX 源码依赖，四通道与八分支 ABI、packed INT4、A16、KV8、FP32 GDN、原生 tokenizer、取消／卸载及缓存生命周期。
- Qwen 算子探索：grouped／row-reuse INT4 投影、SwiGLU 融合、GDN gate／conv／prepare 融合、shared seed、deferred／fixed history、静态图及调度／编译器实验。接受和拒绝的候选都保留，不表示全部默认启用。
- 评测：Diagnostic-800 与公开集合 4786 行的冻结输入、答案、评分器、来源声明及协议；现有各尺寸／训练比例分数和抽查例子。
- 速度证据：固定公共前缀＋变化环境的五模型实验；11 月 16 日 H200 B1–32 与 block128 大 batch 源码／报告。大于八分支使用重复 marker，不冒称新训练的 1024 头模型或独立用户并发。

## 还没有达到“完整训练材料全部公开”的部分

**原始训练语料尚未公开打包。** 本版提供实际源码、来源哈希、选中行号与 packing 清单，不能凭这些清单恢复未提供的原文。不能用 Diagnostic-800 替代训练数据。训练环境也尚未形成经过干净机器重训验证的独立锁定包。

本轮开源家族是上述 Qwen 5% 模型；不是全部历史 checkpoint，Gemma4 以单独的[完整适配器包](../gemma4/README.md)发布，不能与合并全量权重混称。TTS、唱歌、SimpleLove 角色资产属于其他项目，不混入本轮 SimpleTool 发布。

## 本轮实际检查

- 70 项 CPU 测试通过：参考工具客户端、服务协议／content／队列、公开集合适配与评分、中英文诊断评分器。
- 4786 行及原始来源文件哈希验证通过；800 条字面参考答案的评分自洽检查通过。
- 399 份 Python 文件语法检查；shell 入口语法检查；三套 overlay 的相对路径重定位及内容哈希检查通过。
- Swift Package manifest 解析通过。**未在本轮重跑 GPU 或 iPhone 推理，也未执行干净机器完整重训。** 旧速度和真机结果仍保留其实际日期、设备、精度和输入条件。

入口：[README](README.md)、[训练说明](training/README.md)、[端侧内存与数值坑](inference/apple/docs/PITFALLS.md)、[CUDA 实现](inference/cuda/README.md)。
