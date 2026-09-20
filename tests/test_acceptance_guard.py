"""Focused checks for acceptance resource guardrails."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acceptance


class ResourceGuardTests(unittest.TestCase):
    def test_stream_reads_one_final_usage_without_adding_chunks(self):
        client = acceptance.Client.__new__(acceptance.Client)
        client.records = []
        usage = {'prompt_tokens': 100, 'completion_tokens': 2, 'total_tokens': 102,
                 'prompt_tokens_details': {'cached_tokens': 64}}
        def event(data):
            return b'data: ' + json.dumps(data).encode() + b'\n'
        lines = [event({'choices': [{'delta': {'content': 'Hello'}}], 'usage': None}),
                 event({'choices': [{'delta': {'content': ' world'}}], 'usage': None}),
                 event({'choices': [], 'usage': usage}), b'data: [DONE]\n']
        client.open = lambda _path, _body: contextlib.nullcontext(lines)
        with contextlib.redirect_stdout(io.StringIO()):
            result = client.generate('stream-usage', {'max_tokens': 8}, stream=True)
        self.assertEqual(result['usage'], usage)
        self.assertEqual(result['choices'][0]['message']['content'], 'Hello world')
        self.assertEqual(client.records[0]['total_tokens'], 102)
        self.assertEqual(client.records[0]['cached_prompt_tokens'], 64)
        client.open = lambda _path, _body: contextlib.nullcontext(lines[:-1] + lines[-2:])
        with self.assertRaisesRegex(AssertionError, 'one final usage event'):
            client.generate('duplicate-usage', {'max_tokens': 8}, stream=True)

    def test_usage_distinguishes_missing_null_zero_and_hit(self):
        base = {'prompt_tokens': 100, 'completion_tokens': 4, 'total_tokens': 104}
        self.assertIsNone(acceptance.checked_usage(base))
        self.assertIsNone(acceptance.checked_usage(base | {'prompt_tokens_details': None}))
        self.assertIsNone(acceptance.checked_usage(base | {'prompt_tokens_details':
                                                          {'cached_tokens': None}}))
        self.assertEqual(acceptance.checked_usage(base | {'prompt_tokens_details':
                                                          {'cached_tokens': 0}}), 0)
        self.assertEqual(acceptance.checked_usage(base | {'prompt_tokens_details':
                                                          {'cached_tokens': 64}}), 64)
        for bad in (base | {'total_tokens': 168},
                    base | {'prompt_tokens_details': {'cached_tokens': 101}},
                    base | {'prompt_tokens_details': {'cached_tokens': '0'}}):
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                acceptance.checked_usage(bad)

    def resources(self, *, minimum=17 * 2**30, oom=0, swap_seconds=0):
        return {
            'minimum_mem_available_bytes': minimum,
            'oom_kills': oom,
            'longest_continuous_swap_seconds': swap_seconds,
        }

    def test_brief_swap_activity_is_recorded_without_failing(self):
        self.assertFalse(acceptance.resource_guardrail_failed(
            self.resources(swap_seconds=14), 16 * 2**30))

    def test_sustained_swap_memory_floor_and_oom_each_fail(self):
        floor = 16 * 2**30
        self.assertTrue(acceptance.resource_guardrail_failed(
            self.resources(swap_seconds=15), floor))
        self.assertTrue(acceptance.resource_guardrail_failed(
            self.resources(minimum=floor - 1), floor))
        self.assertTrue(acceptance.resource_guardrail_failed(
            self.resources(oom=1), floor))


if __name__ == '__main__':
    unittest.main()
