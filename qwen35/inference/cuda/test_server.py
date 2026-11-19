"""CPU admission/error tests; these do not claim GPU or model acceptance."""
import concurrent.futures,json,queue,sys,tempfile,threading,time,types,unittest
from pathlib import Path
from unittest.mock import patch
from server import Service,handler,ThreadingHTTPServer
import urllib.request,urllib.error

class FakeRuntime:
    def __init__(self,config):self.calls=[]
    def receipt(self):return {'fake_runtime':True}
    def infer(self,body,deadline=None):
        self.calls.append(body);return {'valid':True,'request_id':body.get('request_id'),'tools_executed':False}

class Admission(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.config={'queue_capacity':1,'state_dir':self.temp.name,'initial_scenarios':[],
           'log_outputs':False,'max_body_bytes':2000,'request_timeout_ms':200,'initialize_timeout_ms':1000}
        self.service=Service(self.config)
    def tearDown(self):self.temp.cleanup()
    def test_not_ready_is_503(self):
        code,result=self.service.submit('infer',{},100);self.assertEqual(code,503)
    def test_full_queue_is_429_and_does_not_execute(self):
        self.service.state['status']='ready';self.service.queue.put_nowait(object())
        code,result=self.service.submit('infer',{'request_id':'full'},100)
        self.assertEqual(code,429);self.assertEqual(self.service.queue.qsize(),1)
    def test_single_owner_accepts_and_records_result(self):
        with patch.dict(sys.modules,{'runtime':types.SimpleNamespace(Runtime=FakeRuntime)}):
            thread=threading.Thread(target=self.service.owner,daemon=True);thread.start()
            end=time.monotonic()+2
            while self.service.state['status']!='ready' and time.monotonic()<end:time.sleep(.005)
        code,result=self.service.submit('infer',{'request_id':'cpu-only'},500)
        self.assertEqual(code,200);self.assertFalse(result['tools_executed'])
        self.assertEqual(len(self.service.runtime.calls),1)
        self.assertEqual(json.loads((Path(self.temp.name)/'last-inference.json').read_text())['request_id'],'cpu-only')
    def test_http_readiness_and_json_failure(self):
        http=ThreadingHTTPServer(('127.0.0.1',0),handler(self.service));thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(http.server_port)
        try:
            with urllib.request.urlopen(base+'/healthz') as r:self.assertEqual(r.status,200)
            with self.assertRaises(urllib.error.HTTPError) as cm:urllib.request.urlopen(base+'/readyz')
            self.assertEqual(cm.exception.code,503);cm.exception.close()
            req=urllib.request.Request(base+'/v1/infer',data=b'{"x":1,"x":2}',headers={'Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as cm:urllib.request.urlopen(req)
            self.assertEqual(cm.exception.code,400);cm.exception.close()
        finally:http.shutdown();http.server_close();thread.join()
if __name__=='__main__':unittest.main()
