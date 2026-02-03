<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>
<h1 align="center">SimpleTool</h1>

<p align="center">
  <b>面向实时 LLM 函数调用的并行解码方案</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-论文-red"></a>
  <a href="https://huggingface.co/Cialtion/SimpleTool"><img src="https://img.shields.io/badge/🤗-模型-yellow"></a>
  <a href="https://www.modelscope.cn/models/cialtion/SimpleTool"><img src="https://img.shields.io/badge/ModelScope-模型-blue"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Apache%202.0-green"></a>
</p>

<p align="center">
  基于 LLM 的 4B 模型实现 <b>16Hz 实时函数调用</b>
</p>

---

SimpleTool 通过并行解码实现**实时 LLM 函数调用**。通过引入特殊 token 压缩冗余输出（4-6 倍），并支持函数名与参数的独立生成，我们实现了 **3-6 倍端到端加速**，同时保持了有竞争力的准确率。

<p align="center">
  <img src="assets/fig_title_panel_a.png" alt="SimpleTool 概览" width="700">
</p>

## 工作原理

传统函数调用按顺序生成 token：`function → arg1 → arg2 → ...`，延迟随输出长度线性增长。SimpleTool 利用了两个关键观察：

1. **Token 冗余**：结构化输出包含可预测的 token（括号、参数名），可压缩为特殊 token
2. **弱因果依赖**：函数参数之间基本独立，可以并行生成

<p align="center">
  <img src="assets/overview.png" alt="SimpleTool 架构" width="600">
</p>

通过将函数名和参数解码为共享相同前缀 KV cache 的并行流，延迟从 `sum(token_times)` 变为 `max(head_time)`。并行头利用了内存带宽瓶颈解码阶段的闲置算力，使并行化几乎零开销。

更多细节请参阅我们的 [arXiv 论文](https://arxiv.org/abs/xxxx.xxxxx)。

---

## 快速开始

### 1. 环境配置

```bash
git clone https://github.com/HaxxorCialtion/SimpleTool.git
cd SimpleTool

uv venv env_rt -p python3.12
source env_rt/bin/activate  # Linux

uv pip install -r requirements.txt
```

### 2. 下载模型（AWQ 量化版）

| 模型 | 参数量 | 延迟 | HuggingFace | ModelScope |
|------|--------|------|-------------|------------|
| RT-Qwen2.5-0.5B-AWQ | 0.5B | ~30ms | [🤗](https://huggingface.co/Cialtion/SimpleTool/tree/main/RT-Qwen2.5-0.5B-AWQ) | [链接](https://www.modelscope.cn/models/cialtion/SimpleTool/tree/master/RT-Qwen2.5-0.5B-AWQ) |
| RT-Qwen2.5-1.5B-AWQ | 1.5B | ~40ms | [🤗](https://huggingface.co/Cialtion/SimpleTool/tree/main/RT-Qwen2.5-1.5B-AWQ) | [链接](https://www.modelscope.cn/models/cialtion/SimpleTool/tree/master/RT-Qwen2.5-1.5B-AWQ) |
| RT-Qwen2.5-3B-AWQ | 3B | ~50ms | [🤗](https://huggingface.co/Cialtion/SimpleTool/tree/main/RT-Qwen2.5-3B-AWQ) | [链接](https://www.modelscope.cn/models/cialtion/SimpleTool/tree/master/RT-Qwen2.5-3B-AWQ) |
| RT-Qwen3-4B-AWQ | 4B | ~60ms | [🤗](https://huggingface.co/Cialtion/SimpleTool/tree/main/RT-Qwen3-4B-AWQ) | [链接](https://www.modelscope.cn/models/cialtion/SimpleTool/tree/master/RT-Qwen3-4B-AWQ) |
| RT-Qwen2.5-7B-AWQ | 7B | ~70ms | [🤗](https://huggingface.co/Cialtion/SimpleTool/tree/main/RT-Qwen2.5-7B-AWQ) | [链接](https://www.modelscope.cn/models/cialtion/SimpleTool/tree/master/RT-Qwen2.5-7B-AWQ) |
| RT-Qwen2.5-14B-AWQ | 14B | ~130ms | [🤗](https://huggingface.co/Cialtion/SimpleTool/tree/main/RT-Qwen2.5-14B-AWQ) | [链接](https://www.modelscope.cn/models/cialtion/SimpleTool/tree/master/RT-Qwen2.5-14B-AWQ) |
| RT-Qwen3-30B-A3B-AWQ | 30B-A3B | ~ | [🤗](https://huggingface.co/Cialtion/SimpleTool/tree/main/RT-Qwen3-30B_awq_w4a16) | [链接](https://www.modelscope.cn/models/cialtion/SimpleTool/tree/master/RT-Qwen3-30B_awq_w4a16) |

> 延迟在 RTX 4090 上使用 vLLM prefix caching 测得

```bash
mkdir models
# 使用 huggingface-cli 下载
huggingface-cli download Cialtion/SimpleTool --include "RT-Qwen3-4B-AWQ/*" --local-dir ./models

# 或使用 modelscope 下载
modelscope download --model cialtion/SimpleTool --include "RT-Qwen3-4B-AWQ/*" --local_dir ./models
```

### 3. 启动服务器

编辑 `server/rt_server.py` 中的 `MODEL_PATH`，然后：

```bash
python server/rt_server.py
```

```
╔══════════════════════════════════════════════════════════════════════╗
║          SimpleTool vLLM-Server v1.0                                 ║
║          Having a Realtime LLM based control time!                   ║
║                                                                      ║
║   Run Demos: Open demos/*.html in browser                            ║
║   Build New: Send simpletool-game-guide.md to AI (Claude, Gemini...) ║
║              for building your own HTML games                        ║
╚══════════════════════════════════════════════════════════════════════╝
```

服务器运行在 `http://localhost:8899`。

### 4. 运行演示

在浏览器中打开：

| 演示 | 描述 | 文件 |
|------|------|------|
| **Pong** | AI vs 人类乒乓球游戏 | `demos/pong_game.html` |
| **Neon Arena** | 多 AI 对战射击游戏 | `demos/neon_arena.html` |

对于 Neon Arena 或其他需要额外资源的游戏：
```bash
cd ./demos/neon_arena
python3 -m http.server 8080 --bind 127.0.0.1
```
然后打开 <http://127.0.0.1:8080/neon_arena.html>，输入你的 SimpleTool 服务器地址（默认：`http://localhost:8899`）。来挑战基于 LLM 的 AI 吧！

<p align="center">
  <img src="assets/demo_pong.gif" alt="Pong 演示" width="400">
  <img src="assets/demo_arena.gif" alt="Arena 演示" width="400">
</p>

---

## 构建你自己的游戏

想创建一个新的实时 AI 游戏？将 **`simpletool_game_guide.md`** 作为上下文发送给 AI 编程助手（Claude、GPT 等）。它包含：

- 服务器 API 规范
- 工具定义格式
- Vibe Coding Prompt示例
- 前端代码模板
- 动态头数优化

---

## TODO

- [ ] Windows 原生支持
- [ ] iOS 部署
- [ ] 实时世界模拟
- [ ] 具身数字人

---

## 引用

即将发布

## 许可证

Apache 2.0
