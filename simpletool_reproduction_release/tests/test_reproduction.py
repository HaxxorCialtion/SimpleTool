import unittest
from simpletool_eval.conversion import convert_raw_heads_to_fc
from simpletool_eval.evaluate import aggregate, score_dataset
from simpletool_eval.rules import evaluate_single_prediction, has_tool_exceeding_max_params
from simpletool_eval.data import selected_names
from pathlib import Path


class ReproductionTests(unittest.TestCase):
    def test_argument_order_types_and_null(self):
        tools = [{'name': 'f', 'parameters': {'properties': {'z': {}, 'a': {}, 'optional': {}}}}]
        result = convert_raw_heads_to_fc({'function': 'f', 'arg1': '[1,2]', 'arg2': 'true', 'arg3': '<|null|>'}, tools)
        self.assertEqual(result, [{'name': 'f', 'arguments': {'z': [1, 2], 'a': True}}])

    def test_legacy_optional_alternatives_and_normalization(self):
        gt = [{'f': {'x': ['HELLO World', 'hi'], 'n': [2], 'optional': ['']}}]
        pred = [{'name': 'f', 'arguments': {'x': ' hello  world ', 'n': '2.0'}}]
        self.assertTrue(evaluate_single_prediction(pred, gt)['overall_correct'])
        pred[0]['arguments']['n'] = 3
        self.assertEqual(evaluate_single_prediction(pred, gt), {'func_correct': True, 'overall_correct': False})

    def test_legacy_first_call_extra_args(self):
        pred = [{'name': 'f', 'arguments': {'extra': 9}}, {'name': 'other', 'arguments': {}}]
        self.assertTrue(evaluate_single_prediction(pred, [{'f': {}}])['overall_correct'])

    def test_filter_any_candidate_tool(self):
        item = {'function': [{'name': 'f', 'parameters': {'properties': {}}},
                             {'name': 'other', 'parameters': {'properties': {str(i): {} for i in range(7)}}}]}
        self.assertTrue(has_tool_exceeding_max_params(item))

    def test_coverage_does_not_drop_failures(self):
        rows = [{'id': 'a', 'function': []}, {'id': 'b', 'function': []}]
        gt = {'a': [{'f': {}}], 'b': [{'f': {}}]}
        with self.assertRaises(ValueError):
            score_dataset(rows, gt, [{'id': 'a', 'result': []}])
        scores, _ = score_dataset(rows, gt, [{'id': 'a', 'result': []}, {'id': 'b', 'result': []}])
        self.assertEqual(scores['total'], 2)
        self.assertEqual(scores['overall_acc'], 0)
        with self.assertRaises(ValueError):
            score_dataset(rows, gt, [{'id': 'a', 'result': []}]*2)

    def test_aggregate_keeps_zero_groups(self):
        results = {Path(n).stem: {'func_acc': 0.0, 'overall_acc': 0.0, 'total': 1,
                                'func_correct': 0, 'overall_correct': 0, 'missing_prediction_ids': []}
                   for n in selected_names()}
        results['mobile_actions'].update(func_acc=1.0, overall_acc=1.0, func_correct=1, overall_correct=1)
        scores = aggregate(results)
        self.assertEqual(scores['five_groups']['overall_acc']['macro_average'], 0.2)
        self.assertEqual(scores['five_groups']['overall_acc']['legacy_nonzero_average'], 1.0)
        self.assertAlmostEqual(scores['three_groups']['overall_acc']['macro_average'], 1/3)
        del results['BFCL_v3_simple']
        self.assertIsNone(aggregate(results))

if __name__ == '__main__':
    unittest.main()
