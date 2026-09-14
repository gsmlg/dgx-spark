"""Focused checks for acceptance resource guardrails."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acceptance


class ResourceGuardTests(unittest.TestCase):
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
