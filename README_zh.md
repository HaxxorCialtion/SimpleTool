<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>
<h1 align="center">SimpleTool</h1>

<p align="center">
  <b>面向实时 LLM 函数调用的并行解码架构</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.00030"><img src="https://img.shields.io/badge/arXiv-2603.00030-red"></a>
  <a href="https://icml.cc/Conferences/2026"><img src="https://img.shields.io/badge/ICML%202026-Accepted-4B8BBE"></a>
  <a href="https://huggingface.co/Cialtion/SimpleTool"><img src="https://img.shields.io/badge/🤗-Models-yellow"></a>
  <a href="https://www.modelscope.cn/models/cialtion/SimpleTool"><img src="https://img.shields.io/badge/ModelScope-Models-blue"></a>
  <a href="#演示视频"><img src="https://img.shields.io/badge/Bilibili-Demo-00A1D6?logo=bilibili&logoColor=white"></a>
  <a href="#演示视频"><img src="https://img.shields.io/badge/YouTube-Demo-FF0000?logo=youtube&logoColor=white"></a>
  <a href="#许可证"><img src="https://img.shields.io/badge/License-Apache%202.0-green"></a>
</p>

<p align="center">
  一个 4B 参数的 LLM，实现 <b>16 Hz 端到端实时函数调用</b>——足以驱动游戏 AI、机械臂控制和数字人动画。
</p>

<p align="center">
  <b>最新消息：</b>论文以 <i>RealtimeTool: Parallel Decoding for Real-Time LLM Function Calling</i> 为题被 <b>ICML 2026</b> 接收。本仓库提供 SimpleTool/RealtimeTool 的开源实现、模型与 Demo。
</p>

---

SimpleTool 通过多头并行解码实现**实时 LLM 函数调用**。我们引入特殊 token 来压缩结构化输出中的冗余信息（4–6 倍压缩），并让函数名与各参数独立并行生成，从而实现**端到端 3–6 倍加速**，同时在三大应用场景——**游戏**、**机械臂控制**和**数字人动画**——中保持具有竞争力的准确率。

<p align="center">
  <img src="assets/fig_title_panel_a.png" alt="SimpleTool 概览" width="700">
</p>

## 工作原理

传统函数调用按顺序逐 token 生成——`function → arg1 → arg2 → ...`——延迟随输出长度线性增长。SimpleTool 基于两个关键观察：

1. **Token 冗余**：结构化输出中存在大量可预测的 token（括号、参数名、引号等），可以压缩为单个特殊 token。
2. **弱因果依赖**：函数的各个参数之间基本相互独立，可以并行生成。

<p align="center">
  <img src="assets/overview.png" alt="SimpleTool 架构" width="600">
</p>

将函数名和各参数作为共享同一前缀 KV 缓存的并行流进行解码，延迟从 `sum(所有token耗时)` 降为 `max(单头耗时)`。并行解码头利用了解码阶段显存带宽受限时的闲置算力，使得并行化几乎零开销。

更多细节请参阅我们的 [arXiv 论文](https://arxiv.org/abs/2603.00030)。Camera-ready 论文题为 **RealtimeTool: Parallel Decoding for Real-Time LLM Function Calling**，已被 **ICML 2026** 接收。

---

## 快速上手

### 1. 配置环境

```bash
git clone https://github.com/HaxxorCialtion/SimpleTool.git
cd SimpleTool
```

**方案 A — uv（推荐）**
```bash
uv venv env_rt -p python3.12
source env_rt/bin/activate
uv pip install -r requirements.txt
```

**方案 B — conda**
```bash
conda create -n simpletool python=3.12 -y
conda activate simpletool
pip install -r requirements.txt
```

**方案 C — pip**
```bash
python3.12 -m venv env_rt
source env_rt/bin/activate
pip install -r requirements.txt
```

### 2. 下载模型

默认推荐模型为 **RT-Qwen3-4B-AWQ-v2**（4B 参数，AWQ W4A16 量化，v2 提示格式）。所有脚本默认路径为 `./models/RT-Qwen3-4B-AWQ-v2`。

```bash
# HuggingFace
huggingface-cli download Cialtion/SimpleTool \
  --include "RT-Qwen3-4B-AWQ-v2/*" --local-dir ./models

# 或者 ModelScope（国内推荐）
modelscope download --model cialtion/SimpleTool \
  --include "RT-Qwen3-4B-AWQ-v2/*" --local_dir ./models
```

<details>
<summary><b>全部可用模型</b></summary>

| 模型 | 参数量 | 延迟 | HuggingFace | ModelScope |
|------|--------|------|-------------|------------|
| RT-Qwen2.5-0.5B-AWQ | 0.5B | ~30ms | [🤗](https://huggingface.co/Cialtion/SimpleTool/tree/main/RT-Qwen2.5-0.5B-AWQ) | [链接](https://www.modelscope.cn/models/cialtion/SimpleTool/tree/master/RT-Qwen2.5-0.5B-AWQ) |
| RT-Qwen2.5-1.5B-AWQ | 1.5B | ~40ms | [🤗](https://huggingface.co/Cialtion/SimpleTool/tree/main/RT-Qwen2.5-1.5B-AWQ) | [链接](https://www.modelscope.cn/models/cialtion/SimpleTool/tree/master/RT-Qwen2.5-1.5B-AWQ) |
| RT-Qwen2.5-3B-AWQ | 3B | ~50ms | [🤗](https://huggingface.co/Cialtion/SimpleTool/tree/main/RT-Qwen2.5-3B-AWQ) | [链接](https://www.modelscope.cn/models/cialtion/SimpleTool/tree/master/RT-Qwen2.5-3B-AWQ) |
| **RT-Qwen3-4B-AWQ-v2** | **4B** | **~60ms** | [🤗](https://huggingface.co/Cialtion/SimpleTool/tree/main/RT-Qwen3-4B-AWQ-v2) | [链接](https://www.modelscope.cn/models/cialtion/SimpleTool/tree/master/RT-Qwen3-4B-AWQ-v2) |
| RT-Qwen3-4B-AWQ | 4B | ~60ms | [🤗](https://huggingface.co/Cialtion/SimpleTool/tree/main/RT-Qwen3-4B-AWQ) | [链接](https://www.modelscope.cn/models/cialtion/SimpleTool/tree/master/RT-Qwen3-4B-AWQ) |
| RT-Qwen2.5-7B-AWQ | 7B | ~70ms | [🤗](https://huggingface.co/Cialtion/SimpleTool/tree/main/RT-Qwen2.5-7B-AWQ) | [链接](https://www.modelscope.cn/models/cialtion/SimpleTool/tree/master/RT-Qwen2.5-7B-AWQ) |
| RT-Qwen2.5-14B-AWQ | 14B | ~130ms | [🤗](https://huggingface.co/Cialtion/SimpleTool/tree/main/RT-Qwen2.5-14B-AWQ) | [链接](https://www.modelscope.cn/models/cialtion/SimpleTool/tree/master/RT-Qwen2.5-14B-AWQ) |
| RT-Qwen3-30B-A3B-AWQ | 30B(A3B) | ~ | [🤗](https://huggingface.co/Cialtion/SimpleTool/tree/main/RT-Qwen3-30B_awq_w4a16) | [链接](https://www.modelscope.cn/models/cialtion/SimpleTool/tree/master/RT-Qwen3-30B_awq_w4a16) |

> 延迟数据在 RTX 4090 上使用 vLLM 前缀缓存测得。v2 模型采用改进的更干净的提示词；v1 模型使用之前的多头指令头。

</details>

### 3. 运行基准测试（无需启动服务）

`01_benchmark.py` 通过 vLLM 直接运行多头并行解码，覆盖三大应用场景——游戏 AI、机械臂控制和数字人动画——并输出冷启动 / 热预填充 / 解码瓶颈分析。

```bash
# v2 模型（默认）
python 01_benchmark.py --version v2

# v1 模型
python 01_benchmark.py --version v1 --model ./models/RT-Qwen3-4B-AWQ

# 自动检测每个场景的最优头数
python 01_benchmark.py --n-args auto
```

输出示例：
```
  PARALLEL TEST (v2)

─── Game — Tower Defense ───
PASS  use_skill(Amiya)
  function   use_skill                                     4    OK
  arg1       Amiya                                         4    FILL
  arg2       <|null|>                                      3    NULL
  e2e=24.6ms  max_tok=4

─── Robotic Arm — Assembly ───
PASS  move_to(300,150,50,slow)
  function   move_to                                       4    OK
  arg1       300                                           5    FILL
  arg2       150                                           5    FILL
  arg3       500                                           5    FILL
  arg4       slow                                          3    FILL
  e2e=39.9ms  max_tok=5

─── Digital Human — Streamer ───
PASS  speak(welcome,cheerful)
  function   speak                                         4    OK
  arg1       Welcome!                                      4    FILL
  arg2       cheerful                                      5    FILL
  e2e=29.1ms  max_tok=5

  SUMMARY (v2)
  Accuracy       : 3/3
  Cold start avg : 56.1ms
  Hot prefill avg: 29.3ms
  E2E avg (hot)  : 31.2ms
  E2E / max_tok  : 6.7ms/tok (decode bottleneck)
```

脚本还会打印完整的提示结构和重构后的多头输出，便于检查调试。

### 4. 启动服务

`02_server.py` 将推理引擎封装为 FastAPI 服务，支持 CORS 跨域。HTML 游戏客户端通过它连接模型。

```bash
python 02_server.py
```

服务启动于 `http://localhost:8899`，提供以下接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查，返回模型版本信息 |
| `/v1/function_call` | POST | 多头并行函数调用 |

编辑 `02_server.py` 顶部的 `MODEL_PATH` 和 `MODEL_VERSION` 即可切换 v1/v2 模型。

### 5. 测试服务

服务运行后，在另一个终端中执行：

```bash
python 03_test_server.py
```

该脚本向服务端 API 发送三大场景（游戏、机械臂、数字人）的测试请求，报告准确率、冷启动/热启动延迟及各头输出。

```bash
# 自定义服务地址
python 03_test_server.py --url http://192.168.1.100:8899

# 增加热启动轮数
python 03_test_server.py --rounds 10
```

### 6. 体验 Demo

在浏览器中打开 Demo HTML 文件，它们会连接到正在运行的 SimpleTool 服务。

| Demo | 说明 | 文件 |
|------|------|------|
| **Pong** | AI 对战人类的弹球游戏 | `demos/pong_game.html` |
| **Neon Arena** | 多 AI 对战射击游戏 | `demos/neon_arena.html` |

部分游戏需要额外资源文件：
```bash
cd demos/neon_arena
python3 -m http.server 8080 --bind 127.0.0.1
```
然后打开 http://127.0.0.1:8080/neon_arena.html，输入 SimpleTool 服务地址（默认：`http://localhost:8899`）。

<p align="center">
  <video src="https://github.com/user-attachments/assets/436e3b97-e8ab-4d36-9fa0-8f1962da4a38" autoplay loop muted width="400"></video>
  <video src="https://github.com/user-attachments/assets/f9b127da-b65e-4a06-b48f-836e759a6029" autoplay loop muted width="400"></video>
</p>

---

## 项目结构

```
SimpleTool/
├── 01_benchmark.py          # 第 1 步：直接并行解码基准测试
├── 02_server.py             # 第 2 步：FastAPI vLLM 推理服务
├── 03_test_server.py        # 第 3 步：服务端 API 测试客户端
├── prompts/                 # 外部提示词与场景文件
│   ├── v1_system.txt        #   v1 多头系统提示
│   ├── scenarios.json       #   3 大场景测试用例
│   ├── tools_game.jsonl     #   塔防游戏工具定义
│   ├── tools_arm.jsonl      #   机械臂工具定义
│   └── tools_avatar.jsonl   #   数字人工具定义
├── models/                  # 模型下载目录
│   └── RT-Qwen3-4B-AWQ-v2/ #   默认模型
├── demos/                   # HTML 游戏客户端
│   ├── pong_game.html
│   └── neon_arena/
├── assets/                  # README 配图
├── requirements.txt
├── simpletool-game.skill.md # 用 AI 构建新游戏的指南
├── README.md
└── README_zh.md
```

## 构建你自己的游戏

将 **`simpletool-game.skill.md`** 和本项目的 **`README.md`** 一起喂给你的 AI 编程智能体（Claude Code、Codex、Antigravity 等）即可开始 vibe coding。Skill 文件涵盖服务端 API 规格、工具定义格式、Query 设计最佳实践、前端模板及动态头数优化技巧；README 则帮助 AI 理解整体项目结构。两者配合，即可上手开发基于 SimpleTool 的游戏。

---

## 路线图

- [ ] **世界模拟** — 大规模（1,000+ NPC）实时 AI 异步世界模拟，单智能体行动端到端延迟 < 200ms
- [ ] **推测解码与多 Token 预测** — 引入推测解码（Speculative Decoding）和多 Token 预测，进一步压缩推理延迟
- [ ] **Windows 原生支持** — Windows 游戏引擎插件与原生运行（无需 Docker 或 WSL）
- [ ] **Apple 生态** — Mac 和 iPhone 端侧部署（CoreML / Metal）
- [ ] **v3 架构** — 快思考（实时 SimpleTool）+ 慢思考（异步元认知）融合
- [ ] **具身智能** — 虚拟 3D 数字人，大型游戏引擎集成演示
- [ ] **开源训练** — 完整训练代码与数据集开放

---

## 演示视频

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Bilibili-Demo-00A1D6?logo=bilibili&logoColor=white"></a>
  <a href="#"><img src="https://img.shields.io/badge/YouTube-Demo-FF0000?logo=youtube&logoColor=white"></a>
</p>

> 演示视频即将上线——展示实时游戏 AI、机械臂控制和数字人动画效果。

---

## 引用

```bibtex
@inproceedings{shi2026realtimetool,
  title={RealtimeTool: Parallel Decoding for Real-Time LLM Function Calling},
  author={Shi, Xiaoxin and Wan, Jiaxin and Dong, Linkang and Jiang, Wei and Liu, Yue and Huang, Zengfeng},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML 2026)},
  year={2026}
}
```

## 联系方式

- **邮箱**：cialtion737410@sjtu.edu.cn / cialtion@outlook.com
- **QQ 群**：861244702
- **Bilibili**：[Cialtion](https://space.bilibili.com/Cialtion)

## 许可证

Apache 2.0
