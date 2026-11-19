import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location("gemma_evaluation",Path(__file__).resolve().parents[1]/"source/evaluate_gemma.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class NativePromptTests(unittest.TestCase):
 def test_preserves_bodies_and_roles(self):
  body='工具 {"required":["title","time"]}\n原文保持不变'
  text='<|im_start|>system\n'+body+'<|im_end|>\n<|im_start|>user\n提醒我<|im_end|>\n<|im_start|>assistant\n'
  self.assertEqual(m.native_prompt(text,''),'<bos><|turn>system\n'+body+'<turn|>\n<|turn>user\n提醒我<turn|>\n<|turn>model\n')
 def test_restores_closed_thought_without_content(self):
  text='<|im_start|>assistant\n';thought='<|channel>thought\n<channel|>'
  self.assertEqual(m.native_prompt(text,thought),'<bos><|turn>model\n'+thought)
 def test_rejects_missing_generation_marker(self):
  with self.assertRaises(AssertionError):m.native_prompt('<|im_start|>user\nhello<|im_end|>','')
if __name__=='__main__':unittest.main()
