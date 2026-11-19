import copy
import datetime
import json
import math
import re
import runpy
import unittest
from pathlib import Path
from benchmark_lib import candidate_schemas,input_object,render_prompt,score_case,strict_json,typed_equal,validate_rows
ROOT=Path(__file__).resolve().parents[1]

class EnglishBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[json.loads(l) for l in (ROOT/'cases.jsonl').read_text().splitlines()]
        cls.by_id={r['id']:r for r in cls.rows}

    def row(self,f,n,v='a'):return self.by_id[f'en_{f}_{n:02}_{v}']

    def test_200_rows_100_pairs_and_all_literal_oracles(self):
        self.assertEqual(len(validate_rows(self.rows)),100)
        self.assertEqual(sum(r['contrast']['effect']=='invariant' for r in self.rows),10)
        for row in self.rows:self.assertTrue(score_case(row['expected'],row)['exact_correct'],row['id'])

    def test_english_only_inputs_and_labels(self):
        for r in self.rows:
            self.assertEqual(r['language'],'en')
            self.assertIsNone(re.search('[\u3400-\u9fff]',json.dumps(r,ensure_ascii=False)),r['id'])

    def test_frozen_export(self):
        if not (ROOT/'cases.jsonl').exists():self.skipTest('build first')
        frozen=[strict_json(l) for l in (ROOT/'cases.jsonl').read_text().splitlines()]
        self.assertEqual(len(validate_rows(frozen)),100)
        for r in frozen:
            self.assertTrue(score_case(r['expected'],r)['exact_correct'])
        public=[strict_json(l) for l in (ROOT/'inputs.jsonl').read_text().splitlines()]
        self.assertTrue(all(set(p)=={'id','tools','environment','history','query'} for p in public))

    def test_independent_arithmetic_environment(self):
        for v in ('a','b'):
            r=self.row('environment',4,v);e=r['environment'];a=r['expected']['arguments']
            self.assertEqual(a['offset_bytes'],e['acknowledged_bytes']);self.assertEqual(a['length_bytes'],e['total_bytes']-a['offset_bytes'])
            r=self.row('environment',5,v);e=r['environment'];self.assertEqual(r['expected']['arguments']['x'],(e['x']//e['grid_step'])*e['grid_step'])
            r=self.row('environment',9,v);e=r['environment'];self.assertEqual(r['expected']['arguments']['width_px'],round(e['width_mm']*e['dpi']/25.4))
            r=self.row('environment',10,v);e=r['environment'];a=r['expected']['arguments']
            for k in ('left','top','right','bottom'):self.assertEqual(a[k],e['css_edges'][k]*e['device_pixel_ratio'])
            r=self.row('environment',13,v);e=r['environment'];day=datetime.date.fromisoformat(e['uploaded_date'])+datetime.timedelta(days=e['retention_days'])
            self.assertEqual(r['expected']['arguments']['expires_on'],day.isoformat())
            r=self.row('environment',25,v);e=r['environment'];self.assertEqual(r['expected']['arguments']['deadline_ms'],e['now_ms']+e['timeout_ms'])

    def test_independent_arithmetic_binding(self):
        for v in ('a','b'):
            r=self.row('binding',4,v);e=r['environment'];a=r['expected']['arguments']
            for k in ('left','right'):self.assertEqual(a[k],e['uv'][k]*e['width'])
            for k in ('top','bottom'):self.assertEqual(a[k],e['uv'][k]*e['height'])
            r=self.row('binding',5,v);e=r['environment'];a=r['expected']['arguments']
            self.assertEqual(a['pitch'],{'C4':60,'D4':62}[e['note_name']]+e['transpose_semitones']);self.assertEqual(a['duration_ticks'],e['beats']*e['ticks_per_beat'])
            r=self.row('binding',10,v);e=r['environment'];a=r['expected']['arguments']
            self.assertEqual((datetime.datetime.fromisoformat(a['end'])-datetime.datetime.fromisoformat(a['start'])).total_seconds(),e['minutes']*60)
            r=self.row('binding',16,v);e=r['environment'];a=r['expected']['arguments']
            self.assertEqual(sum(a['counts']),e['units']);self.assertEqual(a['counts'][0],min(e['units'],e['bins'][0]['capacity']))
            r=self.row('binding',19,v);e=r['environment'];a=r['expected']['arguments']
            self.assertEqual(a['left'],e['local_region']['x']+e['window_origin']['x']);self.assertEqual(a['top'],e['local_region']['y']+e['window_origin']['y'])
            r=self.row('binding',20,v);e=r['environment'];a=r['expected']['arguments']
            self.assertEqual(a['start_seconds'],e['start_frame']/e['fps']);self.assertEqual(a['end_seconds'],e['end_frame']/e['fps'])

    def test_independent_arithmetic_semantics(self):
        for v in ('a','b'):
            r=self.row('semantics',7,v);e=r['environment'];self.assertEqual(r['expected']['arguments']['shards'],math.ceil(e['items']/e['capacity_per_shard']))
            r=self.row('semantics',8,v);rate=1 if v=='a' else 2
            self.assertEqual(r['expected']['arguments']['cap_cents'],math.ceil(r['environment']['base_cents']*rate/100))
            r=self.row('semantics',18,v);e=r['environment'];octets=list(e['value_u16'].to_bytes(2,e['byte_order']))
            self.assertEqual(r['expected']['arguments']['octets'],octets)

    def test_false_is_not_zero_and_integer_types(self):
        r=self.row('binding',13);p=copy.deepcopy(r['expected']);p['arguments']['color']=0
        self.assertFalse(score_case(p,r)['schema_valid']);self.assertFalse(typed_equal(False,0))
        p=copy.deepcopy(r['expected']);p['arguments']['copies']=True
        self.assertFalse(score_case(p,r)['schema_valid'])

    def test_sixth_argument_is_scored(self):
        r=self.row('binding',13);p=copy.deepcopy(r['expected']);p['arguments']['tray']='tray-wrong'
        self.assertFalse(score_case(p,r)['exact_correct'])
        heads={ 'function':r['expected']['name'] }
        schema=candidate_schemas(r)[r['expected']['name']]
        for i,(k,s) in enumerate(schema['properties'].items(),1):
            value=r['expected']['arguments'][k];heads[f'arg{i}']=value if s['type']=='string' else json.dumps(value)
        heads['arg7']='irrelevant garbage'
        self.assertTrue(score_case(dict(heads=heads),r)['exact_correct'])
        del heads['arg6'];self.assertFalse(score_case(dict(heads=heads),r)['schema_valid'])

    def test_nested_objects_and_parallel_arrays(self):
        r=self.row('binding',24);p=copy.deepcopy(r['expected']);p['arguments']['sample']['value']='12.5'
        self.assertFalse(score_case(p,r)['schema_valid'])
        r=self.row('binding',7);p=copy.deepcopy(r['expected']);p['arguments']['translated_texts'].reverse()
        self.assertFalse(score_case(p,r)['exact_correct'])

    def test_line_ending_and_case_fidelity(self):
        r=self.row('environment',15);p=copy.deepcopy(r['expected']);p['arguments']['separator']='LF'
        self.assertFalse(score_case(p,r)['exact_correct'])
        r=self.row('semantics',10);p=copy.deepcopy(r['expected']);p['arguments']['profile_name']='Readonly'
        self.assertFalse(score_case(p,r)['exact_correct'])

    def test_invalid_json_and_function(self):
        with self.assertRaises(ValueError):strict_json('{"x":0,"x":1}')
        with self.assertRaises(ValueError):strict_json('Infinity')
        r=self.row('binding',24)
        self.assertFalse(score_case(dict(heads=dict(function=r['expected']['name'],arg1='station-8',arg2='{oops',arg3='2026-11-11T10:00:00Z')),r)['schema_valid'])
        self.assertFalse(score_case(dict(name=[],arguments={}),r)['schema_valid'])

    def test_metadata_not_in_prompt(self):
        r=copy.deepcopy(self.rows[0]);p=render_prompt(r)
        for key in ('expected','tags','objective','rationale','contrast'):r[key]='DO_NOT_LEAK'
        self.assertEqual(render_prompt(r),p);self.assertNotIn('DO_NOT_LEAK',p)

    def test_wrong_but_changed_does_not_count(self):
        a=self.row('environment',16,'a');b=self.row('environment',16,'b')
        pa=dict(name='scan_parcel_label',arguments=dict(label='WRONG-A'));pb=dict(name='scan_parcel_label',arguments=dict(label='WRONG-B'))
        self.assertFalse(typed_equal(pa,pb));self.assertFalse(score_case(pa,a)['exact_correct'] and score_case(pb,b)['exact_correct'])

    def test_no_literal_overlap_with_previous_suite(self):
        report=json.loads((ROOT/'independence_check.json').read_text())
        self.assertEqual(report['exact_query_overlap'],[]);self.assertEqual(report['exact_candidate_schema_overlap_count'],0)
        self.assertEqual(report['expected_function_name_overlap'],[])

if __name__=='__main__':unittest.main(verbosity=2)
