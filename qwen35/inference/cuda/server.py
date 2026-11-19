"""One HTTP replica, one CUDA-owner thread, bounded admission queue."""
import argparse
import concurrent.futures
import json
import queue
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from protocol import Invalid,require,number,strict_json

class Service:
    def __init__(self,config):
        self.config=config;self.queue=queue.Queue(maxsize=config['queue_capacity'])
        self.state={'status':'loading','started_unix':time.time(),'active':None,'completed':0,'failed':0}
        self.logdir=Path(config['state_dir']);self.logdir.mkdir(parents=True,exist_ok=True)
    def snapshot(self):return dict(self.state,queue_depth=self.queue.qsize(),queue_capacity=self.queue.maxsize)
    def save(self,name,value):
        p=self.logdir/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
    def owner(self):
        try:
            from runtime import Runtime
            self.runtime=Runtime(self.config)
            self.state['status']='initializing'
            for path in self.config['initial_scenarios']:
                result=self.runtime.add_scenario(strict_json(Path(path).read_text()))
                self.save('scenario-'+result['id']+'.json',result)
            self.save('startup-receipt.json',self.runtime.receipt())
            self.state['status']='ready';self.state['ready_unix']=time.time();self.save('status.json',self.snapshot())
            print('SIMPLETOOL_READY',json.dumps(self.snapshot()),flush=True)
        except BaseException as e:
            self.state.update(status='failed',error=type(e).__name__+': '+str(e));self.save('status.json',self.snapshot());traceback.print_exc();return
        while True:
            task=self.queue.get();future,op,body,arrived,deadline=task
            if not future.set_running_or_notify_cancel():self.queue.task_done();continue
            self.state['active']={'operation':op,'request_id':body.get('request_id'),'started_unix':time.time()}
            try:
                require(time.monotonic()<deadline,'queue deadline expired before execution')
                started=time.perf_counter();wait_ms=(time.monotonic()-arrived)*1000
                if op=='infer':result=self.runtime.infer(body,deadline=deadline)
                elif op=='create':result=self.runtime.add_scenario(body)
                elif op=='list':result={'scenarios':self.runtime.list_scenarios()}
                elif op=='delete':result=self.runtime.remove_scenario(body['id'])
                elif op=='metrics':result=self.runtime.receipt()
                elif op=='reset':
                    require(not self.runtime.scenarios,'delete scenarios before resetting their pinned cache')
                    result={'reset':self.runtime.eng.reset_prefix_cache() is not False}
                else:raise Invalid('unknown operation')
                elapsed=(time.perf_counter()-started)*1000
                result['queue_wait_ms']=wait_ms;result['handler_work_ms']=elapsed
                if op=='infer':
                    if self.config['log_outputs']:
                        with (self.logdir/'requests.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
                    self.save('last-inference.json',result)
                future.set_result(result);self.state['completed']+=1
            except BaseException as e:
                self.state['failed']+=1;future.set_exception(e)
                if not isinstance(e,(Invalid,TimeoutError)):traceback.print_exc()
                # Unknown engine assertions/OOM are not safe to reuse silently.
                if not isinstance(e,(Invalid,TimeoutError)):
                    self.state.update(status='failed',error=type(e).__name__+': '+str(e))
                    self.save('status.json',self.snapshot());self.queue.task_done();return
            finally:self.state['active']=None
            self.queue.task_done()
    def submit(self,op,body,timeout_ms):
        if self.state['status']!='ready':return 503,{'error':'model_not_ready',**self.snapshot()}
        future=concurrent.futures.Future();now=time.monotonic();deadline=now+timeout_ms/1000
        try:self.queue.put_nowait((future,op,body,now,deadline))
        except queue.Full:return 429,{'error':'queue_full','queue_capacity':self.queue.maxsize}
        try:return 200,future.result(timeout=timeout_ms/1000+1)
        except concurrent.futures.TimeoutError:
            future.cancel();return 504,{'error':'deadline_exceeded','request_id':body.get('request_id'),'note':'do not retry an ambiguous active request blindly; inspect its request_id'}
        except TimeoutError as e:return 504,{'error':str(e),'request_id':body.get('request_id')}
        except Invalid as e:return 400,{'error':str(e)}
        except BaseException as e:return 500,{'error':type(e).__name__+': '+str(e)}

def handler(service):
    class Handler(BaseHTTPRequestHandler):
        server_version='SimpleTool/0.1'
        def send_json(self,status,obj):
            raw=json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
            try:
                self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
                self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):self.route('GET')
        def do_POST(self):self.route('POST')
        def do_DELETE(self):self.route('DELETE')
        def route(self,method):
            self.connection.settimeout(15)
            try:
                path=self.path.split('?',1)[0]
                if path=='/healthz':return self.send_json(200,service.snapshot())
                if path=='/readyz':return self.send_json(200 if service.state['status']=='ready' else 503,service.snapshot())
                if path=='/v1/config' and method=='GET':
                    return self.send_json(200,{'engine':service.config,'state':service.snapshot(),'precision':'FP8 weights, BF16 compute, KV/GDN auto; not KV8',
                      'restart_required':['model','tp','max_model_len','max_num_seqs','kv_gib','decode_capture_sizes','max_prefill_graphs','gdn_backend','moe_backend','queue_capacity'],
                      'scenario_initialization':['system','policy','tools','suffix_tokens','warmup_observations','stop_unused_heads','max_tokens','head_max_tokens','temperature','top_p','top_k','seed'],
                      'request_options':{'include_content':{'type':'boolean','default':False},'head_max_tokens':{'content':'optional per-head token budget, 1..1024'}},
                      'vision_enabled':False,'tool_execution_enabled':False})
                body={}
                if method=='POST':
                    length=int(self.headers.get('Content-Length','0'))
                    require(0<length<=service.config['max_body_bytes'],'invalid request body length')
                    body=strict_json(self.rfile.read(length));require(type(body) is dict,'JSON object required')
                op=None
                if path=='/v1/scenarios':op='create' if method=='POST' else 'list' if method=='GET' else None
                elif path.startswith('/v1/scenarios/') and method=='DELETE':op='delete';body={'id':path.rsplit('/',1)[1]}
                elif path in ('/v1/infer','/v1/function_call') and method=='POST':op='infer'
                elif path=='/v1/metrics' and method=='GET':op='metrics'
                elif path=='/v1/cache/reset' and method=='POST':op='reset'
                if op is None:return self.send_json(404,{'error':'unknown_route'})
                timeout=body.get('timeout_ms',service.config['request_timeout_ms'] if op=='infer' else service.config['initialize_timeout_ms'])
                number(timeout,1,600000,'timeout_ms',True)
                self.send_json(*service.submit(op,body,timeout))
            except (ValueError,TypeError,KeyError) as e:self.send_json(400,{'error':str(e)})
        def log_message(self,fmt,*args):
            if service.config.get('http_access_log',False):print('HTTP',fmt%args,flush=True)
    return Handler

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    config=strict_json(Path(a.config).read_text())
    # Engine fields intentionally cannot be changed by an inference POST.
    require(config['tp']==4,'this deployment reserves four GPUs')
    require(config['kv_cache_dtype']=='auto' and config['mamba_ssm_cache_dtype']=='auto','unvalidated cache precision')
    require(1<=config['max_scenarios']<=16 and 1<=config['max_prefill_graphs']<=64,'invalid bounded cache limits')
    require(1<=config['queue_capacity']<=128,'invalid queue_capacity')
    service=Service(config);http=ThreadingHTTPServer((config['host'],config['port']),handler(service))
    # vLLM multiprocessing and signal setup belong to the main/CUDA-owner thread.
    thread=threading.Thread(target=http.serve_forever,daemon=True,name='http-dispatch')
    thread.start();print('HTTP_LISTENING',config['port'],flush=True)
    service.owner();http.shutdown();raise SystemExit(1)

if __name__=='__main__':main()
