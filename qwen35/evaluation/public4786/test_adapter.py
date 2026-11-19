import copy,json,unittest
from benchlib import normalize_tools,build_prompt,score_row
class AdapterTests(unittest.TestCase):
 def tools(self):return [{'name':'f','parameters':{'type':'dict','properties':{'optional':{'type':'string'},'z':{'type':'integer'},'a':{'type':'string'}},'required':['a','z']}}]
 def case(self):
  tools,_=normalize_tools(self.tools())
  return dict(uid='d/1',id='1',dataset='d',tools=tools,ground_truth=[{'f':{'z':[2],'a':['HELLO'],'optional':['']}}])
 def raw(self,**kwargs):
  bodies=dict(function='f',arg1='2',arg2='hello',arg3='<|null|>',arg4='unused garbage');bodies.update(kwargs)
  return dict(heads=[dict(name=k,body=v,stop='closing_marker') for k,v in bodies.items()])
 def test_owner_abi_required_first_without_mutating_source(self):
  source=self.tools();saved=copy.deepcopy(source);tools,changes=normalize_tools(source)
  self.assertEqual(source,saved)
  self.assertEqual(list(tools[0]['function']['parameters']['properties']),['z','a','optional'])
  self.assertEqual(tools[0]['function']['parameters']['required'],['z','a'])
  self.assertTrue(changes)
 def test_unused_heads_and_optional_do_not_gate_required(self):
  r=score_row(self.raw(arg3='extra'),self.case())
  self.assertTrue(r['required_arguments_match']);self.assertFalse(r['full_local_match'])
 def test_wrong_function_and_missing_required_fail(self):
  self.assertFalse(score_row(self.raw(function='other'),self.case())['required_arguments_match'])
  self.assertFalse(score_row(self.raw(arg1='<|null|>'),self.case())['required_arguments_match'])
 def test_boolean_numeric_coercion_is_separate_typed_diagnostic(self):
  c=self.case();c['ground_truth'][0]['f']['z']=[1]
  r=score_row(self.raw(arg1='true'),c)
  self.assertTrue(r['required_arguments_match']);self.assertFalse(r['typed_required_arguments_match'])
 def test_actual_environment_preserved_and_gt_absent(self):
  c=self.case();msgs=[dict(role='user',content='environment: [sunny]\nhistory: []\nquery')]
  prompt=build_prompt(msgs,c['tools'])
  self.assertIn(msgs[0]['content'],prompt);self.assertTrue(prompt.endswith('<|im_start|>assistant\n'))
  self.assertNotIn('HELLO',prompt)
if __name__=='__main__':unittest.main()
