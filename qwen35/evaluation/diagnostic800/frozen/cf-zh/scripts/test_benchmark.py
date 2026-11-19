import copy
import json
import unittest
from pathlib import Path
from benchmark_lib import candidate_schemas, normalize_prediction, render_prompt, score_case, strict_json, typed_equal, validate_rows, value_errors

ROOT=Path(__file__).resolve().parents[1]

class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[json.loads(l) for l in (ROOT/'cases.jsonl').read_text().splitlines()]
        cls.by_id={r['id']:r for r in cls.rows}

    def test_dataset_and_all_literal_oracles(self):
        self.assertEqual(len(validate_rows(self.rows)),100)
        for row in self.rows:
            self.assertTrue(score_case(row['expected'],row)['exact_correct'],row['id'])

    def test_frozen_dataset_and_public_inputs(self):
        if not (ROOT/'cases.jsonl').exists():self.skipTest('run build first')
        frozen=[strict_json(l) for l in (ROOT/'cases.jsonl').read_text().splitlines()]
        self.assertEqual(len(validate_rows(frozen)),100)
        public=[strict_json(l) for l in (ROOT/'inputs.jsonl').read_text().splitlines()]
        for r,p in zip(frozen,public):
            self.assertEqual(set(p),{'id','tools','environment','history','query'})
            self.assertEqual(p['tools'],r['tools'])

    def test_invalid_function_type(self):
        row=self.by_id['sem_12_a']
        self.assertFalse(score_case(dict(name=[],arguments={}),row)['schema_valid'])

    def test_boolean_not_integer(self):
        row=self.by_id['sem_12_a'];p=copy.deepcopy(row['expected']);p['arguments']['include_health']=0
        s=score_case(p,row);self.assertFalse(s['required_correct']);self.assertFalse(s['schema_valid'])
        self.assertFalse(typed_equal(False,0));self.assertFalse(typed_equal(True,1))

    def test_nested_object_typed_and_order_independent(self):
        row=self.by_id['binding_15_a'];p=copy.deepcopy(row['expected'])
        p['arguments']['seat']=dict(reversed(list(p['arguments']['seat'].items())))
        self.assertTrue(score_case(p,row)['exact_correct'])
        p['arguments']['seat']['row']='3'
        self.assertFalse(score_case(p,row)['schema_valid'])

    def test_parallel_arrays_do_not_sort_independently(self):
        row=self.by_id['binding_01_a'];p=copy.deepcopy(row['expected']);p['arguments']['quantities'].reverse()
        self.assertFalse(score_case(p,row)['exact_correct'])

    def test_exact_case_and_whitespace(self):
        row=self.by_id['sem_11_a'];p=copy.deepcopy(row['expected']);p['arguments']['resource_id']='ab7'
        self.assertFalse(score_case(p,row)['exact_correct'])
        row=self.by_id['sem_17_a'];p=copy.deepcopy(row['expected']);p['arguments']['text']=p['arguments']['text'].strip()
        self.assertFalse(score_case(p,row)['exact_correct'])

    def test_head_adapter_ignores_unused_tail(self):
        row=self.by_id['sem_12_a']
        p=dict(heads=dict(function='get_battery_info',arg1='false',arg2='BROKEN JSON THAT IS UNUSED',content='ignored'))
        self.assertTrue(score_case(p,row)['exact_correct'])

    def test_bad_structured_head_json_and_duplicate_keys(self):
        row=self.by_id['binding_15_a'];fn=row['expected']['name'];seat=row['expected']['arguments']['seat']
        p=dict(heads=dict(function=fn,arg1='show-2',arg2=json.dumps(seat)))
        self.assertTrue(score_case(p,row)['exact_correct'])
        p['heads']['arg2']='{bad json}'
        self.assertFalse(score_case(p,row)['schema_valid'])
        with self.assertRaises(ValueError):strict_json('{"x":1,"x":2}')
        with self.assertRaises(ValueError):strict_json('NaN')

    def test_missing_required_head(self):
        row=self.by_id['sem_12_a'];p=dict(heads=dict(function='get_battery_info'))
        self.assertFalse(score_case(p,row)['schema_valid'])

    def test_optional_omission_policy(self):
        row=copy.deepcopy(self.by_id['sem_12_a'])
        schema=candidate_schemas(row)['get_battery_info'];schema['required']=[]
        row['expected']['arguments']={}
        p=dict(heads=dict(function='get_battery_info',arg1='<|null|>'))
        self.assertTrue(score_case(p,row)['exact_correct'])
        p['heads']['arg1']='false';self.assertFalse(score_case(p,row)['exact_correct'])

    def test_both_wrong_but_changed_is_not_pair_success(self):
        a=self.by_id['sem_11_a'];b=self.by_id['sem_11_b']
        pa=dict(name='open_resource',arguments=dict(resource_id='wrong-A'))
        pb=dict(name='open_resource',arguments=dict(resource_id='wrong-B'))
        self.assertFalse(typed_equal(pa,pb))
        self.assertFalse(score_case(pa,a)['exact_correct'] and score_case(pb,b)['exact_correct'])

    def test_prompt_metadata_cannot_leak(self):
        row=copy.deepcopy(self.rows[0]);prompt=render_prompt(row)
        row['expected']='DO_NOT_LEAK';row['objective']='DO_NOT_LEAK';row['tags']=['DO_NOT_LEAK']
        self.assertEqual(prompt,render_prompt(row));self.assertNotIn('DO_NOT_LEAK',render_prompt(row))

if __name__=='__main__':unittest.main(verbosity=2)
