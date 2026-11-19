"""Scheduler/response contract tests with a fake engine; no model quality claims."""
import sys,types,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'infra'))
from protocol import ALL_HEADS,Invalid,scenario
from runtime import Runtime

class Tokenizer:
 def __init__(self):
  self.vocab={i+100:'</'+h+'>' for i,h in enumerate(ALL_HEADS)}
  self.vocab.update({1:'wait',2:'record',3:'7',4:'A',5:'Brief ',6:'explanation',7:'.'})
 def decode(self,tokens,**kwargs):return ''.join(self.vocab[i] for i in tokens)

class Engine:
 def __init__(self,r,fn):self.r=r;self.fn=fn;self.active={};self.aborted=[];self.added=[]
 def add_request(self,rid,prompt,params):
  h=rid.rsplit('-',1)[1];end=self.r.ends[h]
  seq=({'function':[1 if self.fn=='wait' else 2]+end,'arg1':[3]+end,
      'arg2':[4,4,4,4]+end,'content':[5,6,7]+end})[h]
  self.active[rid]={'h':h,'seq':seq[:params['max_tokens']],'n':0}
  self.added.append(h)
 def has_unfinished_requests(self):return bool(self.active)
 def abort_request(self,ids):
  for rid in ids:
   if rid in self.active:self.aborted.append(self.active.pop(rid)['h'])
 def step(self):
  out=[]
  for rid,a in list(self.active.items()):
   a['n']+=1;tokens=a['seq'][:a['n']];done=a['n']>=len(a['seq'])
   v=types.SimpleNamespace(token_ids=tokens,finish_reason='stop' if tokens[-1] in self.r.T.vocab and self.r.T.vocab[tokens[-1]].startswith('</') else 'length',stop_reason=None)
   out.append(types.SimpleNamespace(request_id=rid,outputs=[v],finished=done,num_cached_tokens=32))
   if done:self.active.pop(rid)
  return out

def runtime(fn='record',max_model_len=4096):
 r=Runtime.__new__(Runtime)
 tools=[{'type':'function','function':{'name':'record','parameters':{'type':'object','properties':{'value':{'type':'integer'},'label':{'type':'string'}},'required':['value','label']}}},
  {'type':'function','function':{'name':'wait','parameters':{'type':'object','properties':{},'required':[]}}}]
 s=scenario({'id':'test','tools':tools,'max_tokens':16});s.update(prefix_ids=[0]*5,cache_salt='test')
 r.scenarios={'test':s};r.config={'max_model_len':max_model_len,'max_num_seqs':8}
 r.T=Tokenizer();r.markers={h:[i+200] for i,h in enumerate(ALL_HEADS)};r.ends={h:[i+100] for i,h in enumerate(ALL_HEADS)}
 r.rt=types.SimpleNamespace(EVENTS=[]);r.SP=lambda **kw:kw;r.OK=types.SimpleNamespace(CUMULATIVE='cumulative')
 r.metrics={'requests':0,'errors':0};r.n=0;r.eng=Engine(r,fn)
 r._input=lambda spec,body:([0]*32,27,0)
 r._prime=lambda *args:5
 r._cleanup=lambda group,ids:r.eng.abort_request(ids)
 return r

class Content(unittest.TestCase):
 def test_default_does_not_schedule_content(self):
  r=runtime();x=r.infer({'scenario_id':'test'})
  self.assertTrue(x['valid']);self.assertNotIn('content',r.eng.added)
  self.assertIsNone(x['content']);self.assertIsNone(x['content_valid']);self.assertIsNone(x['timings_ms']['content_finished'])
 def test_zero_argument_tool_never_cancels_requested_content(self):
  r=runtime('wait');x=r.infer({'scenario_id':'test','include_content':True})
  self.assertEqual(x['call'],{'name':'wait','arguments':{}})
  self.assertIn('arg2',x['cancelled_heads']);self.assertNotIn('content',r.eng.aborted)
  self.assertEqual(x['content'],'Brief explanation.');self.assertTrue(x['content_valid'])
  self.assertLess(x['timings_ms']['tools_ready'],x['timings_ms']['content_finished'])
 def test_content_truncation_does_not_invalidate_tools(self):
  r=runtime();x=r.infer({'scenario_id':'test','include_content':True,'head_max_tokens':{'content':1}})
  self.assertTrue(x['valid']);self.assertFalse(x['content_valid']);self.assertTrue(x['content_errors'])
  self.assertFalse(x['heads']['content']['closed']);self.assertIsNone(x['timings_ms']['content_ready'])
 def test_toggle_is_per_request(self):
  r=runtime();on=r.infer({'scenario_id':'test','include_content':True});off=r.infer({'scenario_id':'test'})
  self.assertTrue(on['content_generated']);self.assertFalse(off['content_generated'])
  self.assertEqual(on['call'],off['call']);self.assertNotIn('content',r.scenarios['test']['heads'])
 def test_capacity_and_context_reject_before_scheduling(self):
  r=runtime();r.config['max_num_seqs']=3
  with self.assertRaises(Invalid):r.infer({'scenario_id':'test','include_content':True})
  self.assertFalse(r.eng.added)
  r=runtime(max_model_len=50)
  with self.assertRaises(Invalid):r.infer({'scenario_id':'test','include_content':True,'head_max_tokens':{'content':32}})
  self.assertFalse(r.eng.added)
  # An inactive content budget must not enlarge a tool-only request.
  self.assertTrue(r.infer({'scenario_id':'test','head_max_tokens':{'content':32}})['valid'])
if __name__=='__main__':unittest.main()
