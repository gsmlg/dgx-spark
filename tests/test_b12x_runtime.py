"""B12X admission, built-in MTP, and profile-local container exceptions."""
from copy import deepcopy
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
import lifecycle as lc


class B12XRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.profile = lc.profiles.load(lc.ROOT, 'qwen38-flash-next-fp4-local-b12x')
        self.native = deepcopy(self.profile['native'])
        self.metadata = deepcopy(self.profile['metadata'])
        self.host = {'BIND_HOST': '127.0.0.1', 'PORT': 8000, 'SHUTDOWN_TIMEOUT': 300}

    def test_probe_checks_quantization_registry_when_help_omits_choices(self):
        help_text = ' '.join('--' + key for key in self.native.keys() |
                             {'host', 'port', 'shutdown-timeout'}) + '\nqwen3 qwen3_xml'
        command = lc.runtime_vllm.probe_command('image@sha256:digest')
        for methods in (['modelopt_mixed'], []):
            with self.subTest(methods=methods):
                modules = {
                    'vllm.model_executor.model_loader': SimpleNamespace(
                        LoadFormats=SimpleNamespace(__args__=('auto', 'b12x'))),
                    'vllm.v1.attention.backends.registry': SimpleNamespace(
                        AttentionBackendEnum=[SimpleNamespace(name='FLASH_ATTN')]),
                    'vllm.model_executor.layers.quantization': SimpleNamespace(
                        QUANTIZATION_METHODS=methods),
                }
                output = io.StringIO()
                with patch.dict(sys.modules, modules), \
                        patch('subprocess.run', return_value=SimpleNamespace(stdout=help_text)), \
                        contextlib.redirect_stdout(output):
                    exec(command[-1], {})
                if methods:
                    lc.runtime_vllm.validate_help(output.getvalue(), self.native, self.metadata)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'does not provide modelopt_mixed'):
                        lc.runtime_vllm.validate_help(output.getvalue(), self.native, self.metadata)

    def test_builtin_mtp_uses_primary_snapshot_without_auxiliary_model(self):
        lc.runtime_vllm.validate(self.native, self.metadata)
        rendered = lc.runtime_vllm.render(self.native, '/hf-cache/snapshot', self.host)
        config = yaml.safe_load(rendered['config_text'])
        self.assertEqual(config['model'], '/hf-cache/snapshot')
        self.assertEqual(config['speculative-config'],
                         {'method': 'mtp', 'num_speculative_tokens': 4})
        self.assertNotIn('auxiliary-models', self.metadata)
        self.assertEqual(self.native, self.profile['native'])

    def test_mtp_rejects_external_model_unknown_options_and_invalid_counts(self):
        cases = [
            {'method': 'mtp', 'model': 'untracked-draft', 'num_speculative_tokens': 4},
            {'method': 'mtp', 'num_speculative_tokens': True},
            {'method': 'mtp', 'num_speculative_tokens': 0},
            {'method': 'mtp', 'num_speculative_tokens': '4'},
            {'method': 'mtp'},
            {'method': 'unknown', 'num_speculative_tokens': 4},
        ]
        for speculative in cases:
            with self.subTest(speculative=speculative):
                native = dict(self.native, **{'speculative-config': speculative})
                with self.assertRaisesRegex(RuntimeError, 'speculative'):
                    lc.runtime_vllm.validate(native, self.metadata)
        profile = lc.profiles.load(lc.ROOT, 'ornith-1.5-35b-a3b-nvfp4')
        native = dict(profile['native'], **{'speculative-config': self.native['speculative-config']})
        with self.assertRaisesRegex(RuntimeError, 'MTP requires'):
            lc.runtime_vllm.validate(native, profile['metadata'])

    def test_b12x_requires_compatible_kernels_precision_and_checkpoint_disk_ple(self):
        for key, value in (('load-format', 'safetensors'), ('moe-backend', 'auto'),
                           ('pipeline-parallel-size', 2), ('kv-cache-dtype', 'auto'),
                           ('enable-flashinfer-autotune', True)):
            with self.subTest(key=key):
                native = dict(self.native, **{key: value})
                with self.assertRaises(RuntimeError):
                    lc.runtime_vllm.validate(native, self.metadata)
        for key, value in (('VLLM_PLE_TABLE_MEMORY', 'shared'),
                           ('B12X_DISK_BACKEND', 'unknown')):
            with self.subTest(key=key):
                metadata = deepcopy(self.metadata)
                metadata['runtime-environment'][key] = value
                with self.assertRaisesRegex(RuntimeError, 'checkpoint-backed disk PLE'):
                    lc.runtime_vllm.validate(self.native, metadata)
        self.metadata['derived-cache'] = {'kind': 'ple', 'mount': '/ple-cache',
                                          'minimum-free-gib': 64, 'reset-before-start': True}
        with self.assertRaisesRegex(RuntimeError, 'checkpoint-backed disk PLE'):
            lc.runtime_vllm.validate(self.native, self.metadata)

    def test_container_exceptions_do_not_affect_other_profiles(self):
        for row in lc.profiles.list_profiles(lc.ROOT):
            profile = lc.profiles.load(lc.ROOT, row['id'])
            auxiliary = {model['name']: '/hf-cache/' + model['name']
                         for model in row.get('auxiliary-models', [])}
            rendered = lc.ADAPTERS[profile['engine']].render(
                profile['native'], '/hf-cache/snapshot', self.host, auxiliary_models=auxiliary)
            service = yaml.safe_load(lc.render_compose(rendered, row))['services']['llm']
            with self.subTest(profile=row['id']):
                self.assertNotIn('privileged', service)
                if row['id'] == self.metadata['id']:
                    self.assertEqual(service['security_opt'],
                                     ['seccomp=unconfined', 'apparmor=unconfined'])
                    self.assertEqual(service['ulimits']['memlock'], {'soft': -1, 'hard': -1})
                    self.assertFalse(any(v['target'] == '/ple-cache' for v in service['volumes']))
                    self.assertIn({'type': 'bind', 'source': '${RUNTIME_CACHE_DIR:?}',
                                   'target': '/root/.cache'}, service['volumes'])
                    self.assertEqual(service['environment']['TRITON_CACHE_DIR'], '/runtime-cache/triton')
                    self.assertEqual(service['environment']['VLLM_PLE_TABLE_MEMORY'], 'disk')
                    self.assertEqual(service['environment']['B12X_DISK_BACKEND'], 'io_uring')
                    weights = next(v for v in service['volumes'] if v['target'] == '/hf-cache')
                    self.assertTrue(weights['read_only'])
                    self.assertEqual(service['restart'], 'no')
                else:
                    self.assertNotIn('security_opt', service)
                    self.assertNotIn('ulimits', service)
                    self.assertFalse(any(v['target'] == '/root/.cache' for v in service['volumes']))

    def test_profile_storage_budget_includes_large_snapshot_and_host_reserve(self):
        host = {'HF_CACHE': str(lc.ROOT), 'RUNTIME_CACHE': str(lc.ROOT),
                'DISK_RESERVE_GIB': 50, 'PREPARE_OVERHEAD_GIB': 60}
        with patch.object(lc.shutil, 'disk_usage', return_value=SimpleNamespace(free=269 * 2**30)):
            with self.assertRaisesRegex(RuntimeError, 'Insufficient disk reserve'):
                lc.check_disk_capacity(host, self.metadata)
        with patch.object(lc.shutil, 'disk_usage', return_value=SimpleNamespace(free=270 * 2**30)):
            lc.check_disk_capacity(host, self.metadata)

    def test_profile_storage_budget_requires_a_positive_integer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / 'profiles' / self.metadata['id']
            profile.mkdir(parents=True)
            (profile / 'vllm.yaml').write_text(yaml.safe_dump(self.native))
            (profile / 'image.env').write_text('IMAGE=example@sha256:digest\n')
            for value in (True, 0, -1, 160.0, '160'):
                with self.subTest(value=value):
                    metadata = dict(self.metadata, **{'prepare-storage-gib': value})
                    (profile / 'profile.yaml').write_text(yaml.safe_dump(metadata))
                    with self.assertRaisesRegex(RuntimeError, 'prepare-storage-gib'):
                        lc.profiles.load(root, self.metadata['id'])


if __name__ == '__main__':
    unittest.main()
