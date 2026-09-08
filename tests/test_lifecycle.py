"""Focused checks for configuration safety and immutable release handling."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
import lifecycle as lc


class ConfigurationTests(unittest.TestCase):
    def test_env_is_data_and_rejects_unknown_duplicate_and_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'host.env'
            for text in ('BAD=x', 'PORT=1\nPORT=2', 'PORT=$(id)', 'PORT=${OTHER}'):
                path.write_text(text)
                with self.assertRaises(RuntimeError):
                    lc.env_file(path, {'PORT'})
            path.write_text('PORT=8000\n')
            self.assertEqual(lc.env_file(path, {'PORT'}), {'PORT': '8000'})

    def test_yaml_duplicate_rejected(self):
        with self.assertRaises(RuntimeError):
            lc.yaml.load('max-model-len: 262144\nmax-model-len: 8192\n', Loader=lc.UniqueLoader)

    def test_release_cannot_escape_state_root(self):
        for value in ('../../etc', '', 'a' * 40):
            with self.assertRaises(RuntimeError):
                lc.release_path(value)

    def test_atomic_state_is_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'active.json'
            lc.atomic(path, {'active': 'old'})
            lc.atomic(path, {'active': 'new'})
            self.assertEqual(json.loads(path.read_text()), {'active': 'new'})
            self.assertEqual(len(list(path.parent.iterdir())), 1)

    def test_conflict_prevents_gpu_or_port_side_effects(self):
        with patch.object(lc, 'conflicts', return_value=['existing-model']), patch.object(lc, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'existing-model'):
                lc.require_idle({})
            run.assert_not_called()

    def test_failed_candidate_restores_accepted_and_still_fails(self):
        old = {'active': 'old', 'accepted': 'old', 'previous': None, 'running': True}
        with patch.object(lc, 'state', return_value=old), patch.object(lc, 'release'), \
             patch.object(lc, 'conflicts', return_value=[]), \
             patch.object(lc, 'compose'), patch.object(lc, 'stop') as stop, \
             patch.object(lc, 'atomic') as write, patch.object(lc, 'containers', return_value=[]), \
             patch.object(lc, 'launch', side_effect=[RuntimeError('candidate failed'), None]) as launch:
            with self.assertRaisesRegex(RuntimeError, 'candidate failed'):
                lc.replace('new', {})
            self.assertEqual([x.args[0] for x in launch.call_args_list], ['new', 'old'])
            self.assertEqual([x.args[0] for x in stop.call_args_list], ['old', 'new'])
            self.assertEqual(write.call_args.args[1]['active'], 'old')
            self.assertTrue(write.call_args.args[1]['running'])

    def test_invalid_candidate_does_not_stop_active(self):
        old = {'active': 'old', 'accepted': 'old', 'previous': None, 'running': True}
        with patch.object(lc, 'state', return_value=old), \
             patch.object(lc, 'release', side_effect=RuntimeError('bad release')), patch.object(lc, 'stop') as stop:
            with self.assertRaises(RuntimeError):
                lc.replace('bad', {})
            stop.assert_not_called()


if __name__ == '__main__':
    unittest.main()
