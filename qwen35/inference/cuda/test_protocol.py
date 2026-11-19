import copy,sys,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
sys.path[:0]=[str(D/'infra'),str(D/'infra/scenario_v2')]
from protocol import scenario,canonical_tools,assemble,dynamic,necessary,Invalid,request_heads
from make_scenarios import ROBOT
class Contracts(unittest.TestCase):
 def test_property_required_alignment(self):
  tools=copy.deepcopy(ROBOT);s=tools[0]['function']['parameters']
  s['required']=['frame','x_mm'];s['properties']={'speed_mm_s':s['properties']['speed_mm_s'],**{k:v for k,v in s['properties'].items() if k!='speed_mm_s'}}
  t=canonical_tools(tools)[0]['function']['parameters']
  self.assertEqual(list(t['properties']),['x_mm','frame','speed_mm_s','y_mm','z_mm']);self.assertEqual(t['required'],['x_mm','frame'])
 def test_reject_unordered_strict(self):
  tools=copy.deepcopy(ROBOT);tools[0]['function']['parameters']['required'].reverse()
  with self.assertRaises(Invalid):canonical_tools(tools,False)
 def test_max_arity_and_closure_contract(self):
  s=scenario({'id':'a','tools':ROBOT});self.assertEqual(s['heads'],['function','arg1','arg2','arg3','arg4','arg5'])
  self.assertEqual(necessary(s,'finish_task'),['function','arg1']);self.assertEqual(necessary(s,'<|null|>'),['function'])
 def test_typed_arguments_preserve_false_and_zero(self):
  s=scenario({'id':'a','tools':ROBOT});call,errors=assemble(s,{'function':'set_gripper','arg1':'70','arg2':'0','arg3':'junk'})
  self.assertFalse(errors);self.assertEqual(call,{'name':'set_gripper','arguments':{'width_mm':70,'force_n':0}})
  _,errors=assemble(s,{'function':'set_gripper','arg1':'false','arg2':'0'});self.assertTrue(errors)
 def test_optional_null_and_exact_strings(self):
  tools=copy.deepcopy(ROBOT);tools[3]['function']['parameters']['required']=[]
  s=scenario({'id':'a','tools':tools});c,e=assemble(s,{'function':'finish_task','arg1':'<|null|>'});self.assertFalse(e);self.assertEqual(c['arguments'],{})
  c,e=assemble(s,{'function':'finish_task','arg1':' A '});self.assertEqual(c['arguments']['task_id'],' A ')
 def test_invalid_call_does_not_get_coerced(self):
  s=scenario({'id':'a','tools':ROBOT});_,e=assemble(s,{'function':'stop_motion','arg1':'emergency_stop'});self.assertFalse(e)
  _,e=assemble(s,{'function':'does_not_exist','arg1':'123'});self.assertTrue(e)
 def test_observation_and_revision(self):
  s=scenario({'id':'a','tools':ROBOT});t=scenario({'id':'a','tools':ROBOT,'policy':'changed'})
  self.assertNotEqual(s['revision'],t['revision']);self.assertEqual(s['prefix'],scenario({'id':'a','tools':ROBOT})['prefix'])
  self.assertNotEqual(dynamic(s,{'environment':{'sensor':0}}),dynamic(s,{'environment':{'sensor':1}}))
 def test_unrecognized_config_rejected(self):
  with self.assertRaises(Invalid):scenario({'id':'a','tools':ROBOT,'suffix_token':512})
  with self.assertRaises(Invalid):scenario({'id':'a','tools':ROBOT,'temperature':float('nan')})
 def test_content_request_does_not_change_common_prompt(self):
  s=scenario({'id':'a','tools':ROBOT})
  body={'environment':{'sensor':1}}
  self.assertEqual(dynamic(s,body),dynamic(s,dict(body,include_content=True)))
  self.assertEqual(request_heads(s,body),s['heads'])
  self.assertEqual(request_heads(s,dict(body,include_content=True)),s['heads']+['content'])
  for value in ['true',1,None,[]]:
   with self.assertRaises(Invalid):dynamic(s,dict(body,include_content=value))
 def test_content_cap_and_maximum_heads(self):
  tools=copy.deepcopy(ROBOT);schema=tools[0]['function']['parameters']
  schema['properties']['extra']={'type':'integer'};schema['required'].append('extra')
  s=scenario({'id':'a','tools':tools,'head_max_tokens':{'content':96}})
  self.assertEqual(len(request_heads(s,{'include_content':True})),8)
  self.assertNotIn('content',necessary(s,'move_cartesian'))
if __name__=='__main__':unittest.main()
