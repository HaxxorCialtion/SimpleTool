#!/usr/bin/env python3
"""
SimpleTool vLLM Server - Multi-Head Parallel Decoding for Real-Time Function Calling
"""

import json
import time
import os
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from vllm import LLM, SamplingParams

# ==================== Config ====================
MODEL_PATH = "./models/ST-Qwen3-4B-AWQ"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8899
MAX_HISTORY = 6

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ==================== Multi-Head Tags ====================
HEAD_TAGS = ["<content>", "<function>", "<arg1>", "<arg2>", "<arg3>", "<arg4>", "<arg5>", "<arg6>"]
STOP_TOKENS = ["<|null|>", "</content>", "</function>", "</arg1>", "</arg2>", "</arg3>", "</arg4>", "</arg5>", "</arg6>", "<|im_end|>"]

SYSTEM_TEMPLATE = """<|im_start|>system
You are a multi-head parallel function calling model. 
## Output Heads

**Head 0 - <content>**: Natural language response
- Format: <content>response text</content>

**Head 1 - <function>**: Function names to call
- Format: <function>name</function>

**Head 2-7 - <arg1>-<arg6>**: Function arguments by position
- Format: <argN>value</argN> 
- If Unnecessary: <argN><|null|></argN>

## Available Tools:

{tools_json}
<|im_end|>
"""


# ==================== Data Models ====================
class Message(BaseModel):
    role: str
    content: str


class FCRequest(BaseModel):
    messages: List[Message]
    tools: List[Dict[str, Any]]
    environment: Optional[List[str]] = None
    history: Optional[List[str]] = None
    max_tokens: int = 32
    temperature: float = 0.0
    include_content_head: bool = False


class FCResponse(BaseModel):
    success: bool
    function: Optional[str] = None
    args: Dict[str, Any] = {}
    heads: Dict[str, str] = {}
    content: Optional[str] = None
    latency_ms: float = 0
    error: Optional[str] = None


# ==================== SimpleTool Engine ====================
class SimpleToolEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm: Optional[LLM] = None
        self.sampling_params = None

    def initialize(self):
        print(f"[SimpleTool] Loading model: {self.model_path}")
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            enable_prefix_caching=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=4096,
            dtype="auto",
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=32,
            stop=STOP_TOKENS,
            include_stop_str_in_output=True
        )
        print("[SimpleTool] Model loaded!")
        self._warmup()

    def _warmup(self):
        print("[SimpleTool] Warming up...")
        dummy_tools = '[{"type":"function","function":{"name":"test","parameters":{}}}]'
        prompt = SYSTEM_TEMPLATE.format(tools_json=dummy_tools)
        prompt += "<|im_start|>user\nenvironment: []\nhistory: []\n\ntest<|im_end|>\n<|im_start|>assistant\n"
        prompts = [prompt + tag for tag in HEAD_TAGS]
        self.llm.generate(prompts, self.sampling_params)
        print("[SimpleTool] Warmup complete!")

    def _build_tools_json(self, tools: List[Dict]) -> str:
        return "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)

    def _extract_param_info(self, tools: List[Dict]) -> List[str]:
        """Extract parameter names in order from tool definitions."""
        names = []
        for tool in tools:
            func = tool.get("function", {})
            params = func.get("parameters", {}).get("properties", {})
            for name in params.keys():
                if name not in names:
                    names.append(name)
        return names[:6]

    def _get_max_args(self, tools: List[Dict]) -> int:
        """Get max argument count across all tools."""
        max_args = 0
        for tool in tools:
            func = tool.get("function", {})
            params = func.get("parameters", {}).get("properties", {})
            max_args = max(max_args, len(params))
        return min(max_args, 6)

    def call(self, request: FCRequest) -> FCResponse:
        start = time.perf_counter()

        tools_json = self._build_tools_json(request.tools)
        system_prompt = SYSTEM_TEMPLATE.format(tools_json=tools_json)

        env_str = json.dumps(request.environment or [], ensure_ascii=False)
        hist_list = (request.history or [])[-MAX_HISTORY:]
        hist_str = ", ".join(hist_list) if hist_list else ""

        query = ""
        for msg in request.messages:
            if msg.role == "user":
                query = msg.content

        user_turn = f"<|im_start|>user\nenvironment: {env_str}\nhistory: [{hist_str}]\n\n{query}<|im_end|>\n<|im_start|>assistant\n"
        full_prefix = system_prompt + user_turn

        # Dynamic head selection based on max args
        max_args = self._get_max_args(request.tools)
        active_tags = ["<function>"] + [f"<arg{i}>" for i in range(1, max_args + 1)]
        if request.include_content_head:
            active_tags = ["<content>"] + active_tags

        prompts = [full_prefix + tag for tag in active_tags]
        outputs = self.llm.generate(prompts, self.sampling_params)

        latency_ms = (time.perf_counter() - start) * 1000

        # Parse outputs
        heads = {}
        head_names = []
        if request.include_content_head:
            head_names.append("content")
        head_names.append("function")
        head_names.extend([f"arg{i}" for i in range(1, max_args + 1)])

        for i, output in enumerate(outputs):
            text = output.outputs[0].text.strip()
            for stop in STOP_TOKENS:
                if text.endswith(stop):
                    text = text[:-len(stop)].strip()
                    break
            heads[head_names[i]] = text

        func_name = heads.get("function", "").strip()
        if not func_name or func_name == "<|null|>":
            return FCResponse(
                success=False,
                heads=heads,
                content=heads.get("content"),
                latency_ms=latency_ms,
                error="No function called"
            )

        param_names = self._extract_param_info(request.tools)
        args = {}
        for i, name in enumerate(param_names):
            val = heads.get(f"arg{i+1}", "").strip()
            if val and val != "<|null|>":
                if val.isdigit():
                    args[name] = int(val)
                elif val.lstrip('-').replace('.', '', 1).isdigit():
                    args[name] = float(val)
                else:
                    args[name] = val.lower().strip()

        return FCResponse(
            success=True,
            function=func_name,
            args=args,
            heads=heads,
            content=heads.get("content"),
            latency_ms=latency_ms
        )


# ==================== FastAPI ====================
engine: Optional[SimpleToolEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = SimpleToolEngine(MODEL_PATH)
    engine.initialize()
    yield
    print("[Server] Shutdown")


app = FastAPI(title="SimpleTool Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "loaded": engine is not None and engine.llm is not None,
        "model": MODEL_PATH
    }


@app.post("/v1/function_call", response_model=FCResponse)
async def function_call(request: FCRequest):
    if engine is None or engine.llm is None:
        raise HTTPException(503, "Model not loaded")
    try:
        return engine.call(request)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return FCResponse(success=False, error=str(e), latency_ms=0)


if __name__ == "__main__":
    print(r"""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   ███████╗██╗███╗   ███╗██████╗ ██╗     ███████╗                   ║
║   ██╔════╝██║████╗ ████║██╔══██╗██║     ██╔════╝                   ║
║   ███████╗██║██╔████╔██║██████╔╝██║     █████╗                     ║
║   ╚════██║██║██║╚██╔╝██║██╔═══╝ ██║     ██╔══╝                     ║
║   ███████║██║██║ ╚═╝ ██║██║     ███████╗███████╗                   ║
║   ╚══════╝╚═╝╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝                   ║
║                                                                    ║
║          SimpleTool vLLM-Server v1.0                               ║
║          Having a Realtime LLM based control time!                 ║
║                                                                    ║
║   Run Demos: Open demos/*.html in browser                          ║
║   Build New: Send simpletool-game-guide.md to AI(Claude Gemini...) ║
║              for Building new your own HTML games easily           ║                                        ║
║   Endpoints:                                                       ║
║     GET  /health           - Health check                          ║
║     POST /v1/function_call - Function call API                     ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
