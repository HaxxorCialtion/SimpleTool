"""GPU HTTP API. Run python -m vlm_release.remote_api --model CHECKPOINT.

POST /v1/simpletool: optional image_base64 (plain PNG base64), text-only messages,
OpenAI function tools, mode direct|adaptive. Returns prediction/model/gpu/timings/
events. Bearer authentication belongs to the platform gateway. Bind only behind
that gateway. GET /health and /v1/models become ready after both-mode warmup.

The runtime generates function plus arg1..argN, where N is the maximum declared
property count across the request's tools; unused argument branches are omitted.

Request example (image_base64 must contain a real PNG, not a data URL):
  {"image_base64":"...", "messages":[{"role":"user","content":"Read this invoice."}],
   "tools":[{"type":"function","function":{"name":"finish",
             "parameters":{"type":"object","properties":{}}}}], "mode":"direct"}
Successful transport returns HTTP 200 even when prediction.legal is false;
clients must check that flag before executing a tool. Direct omits raw.content.
HTTP 400: invalid contract; 413: body too large; 429: inference queue full;
503: not ready; 500: inference failure. Events are buffered in the response,
not an SSE stream. Timings exclude network transmission and browser execution.
"""
import argparse
import base64
import binascii
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .infer import prepare_input, run_prediction
from .protocol import tool_schemas, SPECIAL

MAX_BODY = 12 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 2048 * 2048


class InputError(ValueError):
    pass


def validate_payload(body):
    from PIL import Image
    if not isinstance(body, dict) or not {'messages', 'tools', 'mode'}.issubset(body) or set(body) - {'image_base64', 'messages', 'tools', 'mode'}:
        raise InputError('Required keys: messages, tools, mode; optional image_base64; no other fields allowed')
    if body['mode'] not in ('direct', 'adaptive'):
        raise InputError('mode must be direct or adaptive')
    messages = body['messages']
    if not isinstance(messages, list) or not 1 <= len(messages) <= 32:
        raise InputError('messages must contain 1..32 text messages')
    total = 0
    for message in messages:
        if not isinstance(message, dict) or set(message) != {'role', 'content'}:
            raise InputError('Each message requires only role and text content')
        if message['role'] not in ('system', 'user', 'assistant') or not isinstance(message['content'], str):
            raise InputError('Only system/user/assistant text messages are accepted; no image URLs or paths')
        if any(token in message['content'] for token in ('<|image_pad|>', '<|video_pad|>', '<|vision_start|>', '<|vision_end|>')):
            raise InputError('Client vision placeholders are forbidden')
        total += len(message['content'])
    if total > 24000 or not any(m['role'] == 'user' for m in messages):
        raise InputError('Text too long or missing user message')
    if not isinstance(body['tools'], list) or len(body['tools']) > 32 or len(json.dumps(body['tools'])) > 24000:
        raise InputError('Too many tools or oversized schemas')
    try:
        tool_schemas(body['tools'])
    except (ValueError, TypeError, KeyError) as exc:
        raise InputError('Invalid tool schema: ' + str(exc)) from None
    value = body.get('image_base64')
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > MAX_IMAGE_BYTES * 4 // 3 + 8:
        raise InputError('Invalid or oversized image_base64')
    try:
        raw = base64.b64decode(value, validate=True)
        if not raw.startswith(b'\x89PNG\r\n\x1a\n') or len(raw) > MAX_IMAGE_BYTES:
            raise InputError('Image must be a PNG of at most 8 MiB')
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != 'PNG' or image.width * image.height > MAX_PIXELS or min(image.size) < 1:
                raise InputError('PNG dimensions exceed 4 megapixels')
            image.verify()
    except (binascii.Error, OSError, ValueError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError('Invalid PNG base64') from None
    return raw


class Runtime:
    def __init__(self, model, queue_size=4):
        self.ready = False
        self.error = None
        self.model_path = str(Path(model).resolve())
        self.identity = {'id': 'corrected_mixed_frozen_r4', 'path': self.model_path,
            'config_sha256': hashlib.sha256((Path(model) / 'config.json').read_bytes()).hexdigest()}
        manifest_path = Path(model) / 'MODEL_SHA256.json'
        if manifest_path.exists():
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            if not isinstance(manifest.get('model_id'), str) or not isinstance(manifest.get('files'), dict):
                raise ValueError('Invalid MODEL_SHA256.json identity manifest')
            self.identity.update(id=manifest['model_id'],
                manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                copied_files_verified=manifest.get('copied_files_verified') is True)
        self.gpu = None
        self.slots = threading.BoundedSemaphore(queue_size)
        self.engine_lock = threading.Lock()

    def load(self, tp=1):
        try:
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams
            import torch
            from PIL import Image
            self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
            self.sampling = SamplingParams
            self.engine = LLM(model=self.model_path, dtype='bfloat16', max_model_len=8192,
                max_num_seqs=16, tensor_parallel_size=tp, gpu_memory_utilization=.65,
                enforce_eager=True, enable_prefix_caching=True, limit_mm_per_prompt={'image': 1})
            self.gpu = {'name': torch.cuda.get_device_name(0), 'tensor_parallel_size': tp}
            stream = io.BytesIO(); Image.new('RGB', (224, 224), 'white').save(stream, format='PNG')
            warm = {'image_base64': base64.b64encode(stream.getvalue()).decode(),
                'messages': [{'role': 'user', 'content': 'Inspect the blank image and call finish.'}],
                'tools': [{'type': 'function', 'function': {'name': 'finish', 'parameters': {'type': 'object', 'properties': {}}}}],
                'mode': 'direct'}
            for mode in ('direct', 'adaptive'):
                warm['mode'] = mode
                result = self.predict(warm, validate_payload(warm), time.perf_counter())
                if not result['prediction'].get('legal'):
                    raise RuntimeError('Warmup generated invalid protocol')
            self.ready = True
        except Exception as exc:
            self.error = type(exc).__name__
            print('API_STARTUP_FAILED', self.error, flush=True)
            traceback.print_exc()

    def predict(self, body, raw, start):
        request_id = uuid.uuid4().hex
        events = []
        def emit(event, data):
            events.append({'event': event, 'data': data, 'elapsed_seconds': time.perf_counter() - start})
        with self.engine_lock:
            queue_seconds = time.perf_counter() - start
            prep = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix='simpletool-api-') as directory:
                messages = [dict(m) for m in body['messages']]
                if raw is not None:
                    image = Path(directory) / 'input.png'; image.write_bytes(raw)
                    index = max(i for i, m in enumerate(messages) if m['role'] == 'user')
                    messages[index]['content'] = [{'type': 'image', 'image': str(image)},
                        {'type': 'text', 'text': messages[index]['content']}]
                    row = {'messages': messages, 'tools': body['tools'], 'image_paths': [str(image)]}
                else:
                    row = {'messages': messages, 'tools': body['tools']}
                inp = prepare_input(self.processor, row, Path(directory)); inp['tools'] = body['tools']
                # The same processor handles text-only rows when no image is supplied.
                encoded = self.processor(text=[inp['prompt']], images=inp['pictures'] or None, return_tensors='pt')
                if encoded['input_ids'].shape[1] + 48 + 64 > 8192:
                    raise InputError('Request exceeds model token budget')
                preparation_seconds = time.perf_counter() - prep
                emit('request_started', {'mode': body['mode']})
                prediction = run_prediction(self.engine, self.sampling, self.processor.tokenizer,
                    inp, body['mode'], 'parallel', 48, 64, event_callback=emit)
                prediction.pop('prefixes', None)  # No server-side temporary paths in response.
                prediction['mode'] = body['mode']
            emit('request_completed', {'legal': prediction.get('legal', False)})
        return {'request_id': request_id, 'prediction': prediction, 'model': self.identity,
            'gpu': self.gpu, 'timings': {'queue_seconds': queue_seconds,
                'preparation_seconds': preparation_seconds, 'total_seconds': time.perf_counter() - start}, 'events': events}


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8
    def __init__(self, address, handler, runtime):
        self.runtime = runtime
        self.connections = threading.BoundedSemaphore(16)
        super().__init__(address, handler)
    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            request.sendall(b'HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
            self.shutdown_request(request)
            return
        request.settimeout(30)
        try:
            super().process_request(request, client_address)
        except Exception:
            self.connections.release()
            raise
    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body):
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw))); self.send_header('Cache-Control', 'no-store')
        self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        runtime = self.server.runtime
        if self.path == '/health':
            self.send(200 if runtime.ready else 503, {'ready': runtime.ready, 'model': runtime.identity,
                'gpu': runtime.gpu, 'startup_error': runtime.error})
        elif self.path == '/v1/models':
            self.send(200 if runtime.ready else 503, {'object': 'list', 'data': [dict(runtime.identity, object='model', ready=runtime.ready)]})
        else:
            self.send(404, {'error': 'not_found'})
    def do_POST(self):
        if self.path != '/v1/simpletool':
            self.send(404, {'error': 'not_found'}); return
        runtime = self.server.runtime
        if not runtime.ready:
            self.send(503, {'error': 'model_not_ready'}); return
        if self.headers.get('Transfer-Encoding'):
            self.send(400, {'error': 'chunked_requests_not_supported'}); return
        try:
            size = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            self.send(400, {'error': 'invalid_content_length'}); return
        if size <= 0 or size > MAX_BODY:
            self.send(413, {'error': 'body_limit_12MiB'}); return
        if not runtime.slots.acquire(blocking=False):
            self.send(429, {'error': 'inference_queue_full'}); return
        start = time.perf_counter()
        try:
            body = json.loads(self.rfile.read(size))
            raw = validate_payload(body)
            result = runtime.predict(body, raw, start)
            self.send(200, result)
        except (InputError, json.JSONDecodeError, UnicodeError) as exc:
            self.send(400, {'error': 'invalid_request', 'detail': str(exc)[:300]})
        except Exception as exc:
            self.send(500, {'error': 'inference_failed', 'type': type(exc).__name__})
        finally:
            runtime.slots.release()
    def log_message(self, fmt, *args):
        print('HTTP', fmt % args, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True); ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000); ap.add_argument('--tensor-parallel-size', type=int, default=1)
    ap.add_argument('--queue-size', type=int, default=4)
    args = ap.parse_args()
    if not 1 <= args.queue_size <= 16:
        ap.error('queue-size must be 1..16')
    runtime = Runtime(args.model, args.queue_size)
    server = Server((args.host, args.port), Handler, runtime)
    # Keep vLLM initialization in the main thread: multiprocessing/signal setup
    # may require it. The HTTP thread serves readiness=503 during model loading.
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    runtime.load(args.tensor_parallel_size)
    if not runtime.ready:
        server.shutdown()
        server.server_close()
        raise SystemExit('GPU API startup failed; inspect traceback')
    server_thread.join()


if __name__ == '__main__':
    main()
