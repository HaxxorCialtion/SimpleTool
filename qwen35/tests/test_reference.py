import importlib.util,json,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parents[1]/'reference/infer_heads.py'
spec=importlib.util.spec_from_file_location('reference',P);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
def tool(name='set_value',properties=None,required=None):
 return dict(type='function',function=dict(name=name,parameters=dict(type='object',properties=properties or {},required=required or [])))
class Response:
 def __init__(self,value):self.value=value
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def read(self):return json.dumps(self.value).encode()
class Reference(unittest.TestCase):
 def infer(self,tools,values,closed=True):
  submitted=[]
  def request(req,timeout):
   data=json.loads(req.data);head=data['stop'][0][2:-1];submitted.append((head,data))
   text=values.get(head,'<|null|>')+('</'+head+'>' if closed else '')
   return Response({'choices':[{'text':text,'finish_reason':'stop' if closed else 'length'}]})
  with patch.object(ref.urllib.request,'urlopen',request):
   out=ref.generate('http://example.invalid/v1','model','PROMPT',tools,'direct')
  return out,submitted
 def test_zero_argument_keeps_null_arg1(self):
  out,requests=self.infer([tool('get_battery')],{'function':'get_battery'})
  self.assertTrue(out['valid']);self.assertEqual(out['arguments'],{});self.assertEqual({x[0] for x in requests},{'function','arg1'})
 def test_required_order_rejected(self):
  with self.assertRaises(AssertionError):ref.build_prompt([tool(properties={'optional':{},'first':{}},required=['first'])],'q')
 def test_prompt_preserves_order(self):
  s=ref.build_prompt([tool(properties={'title':{'type':'string'},'datetime':{'type':'string'}},required=['title','datetime'])],'create')
  self.assertLess(s.index('"title":'),s.index('"datetime":'));self.assertTrue(s.endswith('<|im_start|>assistant\n'))
 def test_candidate_max_and_unused_heads(self):
  ts=[tool('short',{'x':{'type':'integer'}},['x']),tool('long',{f'a{i}':{'type':'string'} for i in range(6)})]
  out,reqs=self.infer(ts,{'function':'short','arg1':'7','arg6':'irrelevant'})
  self.assertTrue(out['valid']);self.assertEqual(out['arguments'],{'x':7});self.assertEqual(len(reqs),7)
 def test_unclosed_is_invalid(self):
  out,_=self.infer([tool()],{'function':'set_value'},False);self.assertFalse(out['valid']);self.assertEqual(out['reason'],'unclosed_head')
 def test_unknown_function_is_invalid(self):
  out,_=self.infer([tool()],{'function':'invented'});self.assertFalse(out['valid']);self.assertEqual(out['reason'],'unknown_function')
 def test_scalar_types_and_null(self):
  ts=[tool(properties={'flag':{'type':'boolean'},'note':{'type':'string'}},required=['flag'])]
  out,_=self.infer(ts,{'function':'set_value','arg1':'false','arg2':'<|null|>'});self.assertIs(out['arguments']['flag'],False);self.assertNotIn('note',out['arguments'])
if __name__=='__main__':unittest.main()
