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
    def test_all_repository_profiles_are_strict_and_renderable(self):
        loaded = {row['id']: lc.profiles.load(lc.ROOT, row['id'])
                  for row in lc.profiles.list_profiles(lc.ROOT)}
        self.assertEqual(set(loaded), {'qwen38-27b-nvfp4', 'laguna-s-2.1-nvfp4',
                                      'qwen38-flash-next-nvfp4'})
        host = {'BIND_HOST': '127.0.0.1', 'PORT': 8000, 'SHUTDOWN_TIMEOUT': 300}
        for profile in loaded.values():
            adapter = lc.ADAPTERS[profile['engine']]
            adapter.validate(profile['native'], profile['metadata'])
            rendered = adapter.render(profile['native'], '/hf-cache/snapshot', host)
            self.assertIn('local-assistant', rendered['resolved'].values())

    def test_profile_resolution_rejects_traversal_and_unknown_ids(self):
        for profile_id in ('../../etc', 'missing-profile', '/tmp'):
            with self.assertRaises(RuntimeError):
                lc.profiles.resolve(lc.ROOT, profile_id)

    def test_sglang_release_uses_argument_vector_and_owned_ple_mount(self):
        profile = lc.profiles.load(lc.ROOT, 'qwen38-flash-next-nvfp4')
        host = {'BIND_HOST': '127.0.0.1', 'PORT': 8000, 'SHUTDOWN_TIMEOUT': 300}
        rendered = lc.runtime_sglang.render(profile['native'], '/hf-cache/snapshot', host,
                                            authenticated=True)
        self.assertEqual(rendered['entrypoint'], ['python3', '-m', 'sglang.launch_server'])
        self.assertIn('--ple-offload-embedding', rendered['command'])
        self.assertIn('--max-total-tokens', rendered['command'])
        self.assertEqual(rendered['resolved']['max-total-tokens'], 32768)
        self.assertIn('${VLLM_API_KEY:?}', rendered['command'])
        compose = lc.render_compose(rendered, profile['metadata'])
        self.assertIn('${PLE_CACHE_DIR:?}', compose)
        self.assertNotIn('/bin/sh', compose)

    def test_derived_cache_reset_is_scoped_to_release_owned_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'ple' / ('a' * 24)
            path.mkdir(parents=True)
            lc.atomic(path / '.owner.json', {'release': 'a' * 24, 'kind': 'ple'})
            (path / 'table.bin').write_bytes(b'derived')
            rec = {'id': 'a' * 24,
                   'identity': {'host': {'RUNTIME_CACHE': str(root)},
                                'profile_policy': {'derived-cache': {'reset-before-start': True}}},
                   'compose_env': {'PLE_CACHE_DIR': str(path)}}
            with patch.object(lc, 'containers', return_value=[]):
                lc.reset_derived_cache(rec)
            self.assertFalse((path / 'table.bin').exists())
            self.assertTrue((path / '.owner.json').exists())

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

    def test_memory_guard_requires_sustained_low_samples(self):
        low_since, failed = lc.sustained_low_memory(None, 15, 16, 100)
        self.assertEqual(low_since, 100)
        self.assertFalse(failed)
        low_since, failed = lc.sustained_low_memory(low_since, 15, 16, 114)
        self.assertFalse(failed)
        self.assertEqual(lc.sustained_low_memory(low_since, 16, 16, 115), (None, False))
        self.assertEqual(lc.sustained_low_memory(low_since, 15, 16, 115), (100, True))

    def test_conflict_prevents_gpu_or_port_side_effects(self):
        with patch.object(lc, 'conflicts', return_value=['existing-model']), patch.object(lc, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'existing-model'):
                lc.require_idle({})
            run.assert_not_called()

    def test_failed_candidate_restores_accepted_and_still_fails(self):
        old = {'active': 'old', 'accepted': 'old', 'previous': None, 'running': True}
        candidate = {'identity': {'host': {'BIND_HOST': '127.0.0.1'}}}
        with patch.object(lc, 'state', return_value=old), patch.object(lc, 'release', return_value=(None, candidate)), \
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

    def test_failed_candidate_restores_running_unaccepted_release(self):
        old = {'active': 'old', 'accepted': None, 'previous': None, 'running': True}
        candidate = {'identity': {'host': {'BIND_HOST': '127.0.0.1'}}}
        with patch.object(lc, 'state', return_value=old), \
             patch.object(lc, 'release', return_value=(None, candidate)), \
             patch.object(lc, 'conflicts', return_value=[]), patch.object(lc, 'compose'), \
             patch.object(lc, 'stop'), patch.object(lc, 'atomic') as write, \
             patch.object(lc, 'launch', side_effect=[RuntimeError('candidate failed'), None]) as launch:
            with self.assertRaisesRegex(RuntimeError, 'candidate failed'):
                lc.replace('new', {})
            self.assertEqual([x.args[0] for x in launch.call_args_list], ['new', 'old'])
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
