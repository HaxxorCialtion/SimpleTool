import unittest
from types import SimpleNamespace
from simpletool_eval.bfcl_latency import build_prompt, heads_for, reset_cache, summarize, timed_question, STOP_STRINGS


class BFCLLatencyTests(unittest.TestCase):
    def test_null_and_every_head_closer_are_stops(self):
        self.assertIn('<|null|>', STOP_STRINGS)
        for head in ['content','function']+[f'arg{i}' for i in range(1,7)]:
            self.assertIn(f'</{head}>', STOP_STRINGS)
        self.assertNotIn('</*>', STOP_STRINGS)

    def test_heads_use_all_candidates_not_answer(self):
        row={'function':[{'name':'one','parameters':{'properties':{'z':{}}}},
                         {'type':'function','function':{'name':'two','parameters':{'properties':{'b':{},'a':{},'c':{}}}}}]}
        self.assertEqual(heads_for(row),['function','arg1','arg2','arg3'])
        row['function'][0]['parameters']['properties']={str(i):{} for i in range(7)}
        with self.assertRaises(ValueError):heads_for(row)

    def test_prompt_preserves_order_and_excludes_answer(self):
        row={'id':'x','question':[[{'role':'user','content':'hello'}]],
             'ground_truth':'secret answer','function':[{'name':'f','parameters':{'properties':{'z':{},'a':{}}}}]}
        for ver in ['v1','v2']:
            prompt=build_prompt(row,ver)
            self.assertNotIn('secret answer',prompt)
            self.assertLess(prompt.index('"z"'),prompt.index('"a"'))
            self.assertTrue(prompt.endswith('<|im_start|>assistant\n'))
        row['question'].append([{'role':'user','content':'second'}])
        with self.assertRaises(ValueError):build_prompt(row,'v1')

    def test_rate_inverse_mean_includes_incorrect_calls(self):
        rows=[{'latency_ms':10,'strict_correct':True,'legacy_correct':True,'function_correct':True,
               'finish_reasons':{'function':'stop'},'token_counts':{'function':2},'head_count':1},
              {'latency_ms':90,'strict_correct':False,'legacy_correct':False,'function_correct':False,
               'finish_reasons':{'function':'length'},'token_counts':{'function':128},'head_count':1}]
        s=summarize(rows)
        self.assertEqual(s['serial_hz'],20)
        self.assertEqual(s['mean_ms'],50)
        self.assertEqual(s['p95_ms'],90)
        self.assertEqual(s['length_limited_calls'],1)
        self.assertEqual(s['strict_correct'],1)

    def test_reset_supports_v1_none_return_and_rejects_false(self):
        calls=[]
        llm=SimpleNamespace(reset_prefix_cache=lambda:calls.append('reset'))
        reset_cache(llm)
        self.assertEqual(calls,['reset'])
        llm.reset_prefix_cache=lambda:False
        with self.assertRaises(RuntimeError):reset_cache(llm)

    def test_generate_is_one_question_at_a_time(self):
        calls=[]
        def generate(prompts, sampling, use_tqdm):
            calls.append(list(prompts))
            return [SimpleNamespace(outputs=[SimpleNamespace(text='ok')]) for _ in prompts]
        llm=SimpleNamespace(generate=generate)
        for q in ['question-a','question-b']:
            outputs,ms=timed_question(llm,[q+'<function>',q+'<arg1>'],None)
            self.assertEqual(len(outputs),2)
            self.assertGreaterEqual(ms,0)
        self.assertEqual(calls,[['question-a<function>','question-a<arg1>'],['question-b<function>','question-b<arg1>']])


if __name__=='__main__':unittest.main()
