"""Focused checks for configuration safety and immutable release handling."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
import lifecycle as lc
from artifact_layout import validate_artifact_layout


class ConfigurationTests(unittest.TestCase):
    def test_artifact_layout_accepts_transformers_and_native_mistral(self):
        layouts = [
            ({'config.json', 'tokenizer_config.json', 'tokenizer.json',
              'model.safetensors.index.json'}, 'model.safetensors.index.json'),
            ({'params.json', 'tokenizer_config.json', 'tokenizer.json', 'tekken.json',
              'chat_template.jinja', 'consolidated.safetensors.index.json'},
             'consolidated.safetensors.index.json'),
        ]
        for required, index_name in layouts:
            with self.subTest(index=index_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                shard = 'weights-00001.safetensors'
                for name in required - {index_name}:
                    (root / name).write_text('{}')
                (root / shard).write_bytes(b'weights')
                (root / index_name).write_text(json.dumps({'weight_map': {'layer': shard}}))
                validate_artifact_layout(root, required | {shard}, 'primary')

    def test_artifact_layout_rejects_incomplete_native_mistral_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {'params.json', 'tokenizer_config.json', 'tokenizer.json',
                     'consolidated.safetensors.index.json', 'weights.safetensors'}
            for name in paths:
                (root / name).write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'model/tokenizer artifacts'):
                validate_artifact_layout(root, paths, 'primary')

    def test_all_repository_profiles_are_strict_and_renderable(self):
        loaded = {row['id']: lc.profiles.load(lc.ROOT, row['id'])
                  for row in lc.profiles.list_profiles(lc.ROOT)}
        self.assertEqual(set(loaded), {'qwen38-27b-nvfp4', 'laguna-s-2.1-nvfp4',
                                      'laguna-xs-2.1-nvfp4', 'qwen38-flash-next-nvfp4',
                                      'gpt-oss-120b-mxfp4', 'muse-glimmer-30b-nvfp4',
                                      'gemma-4-26b-a4b-nvfp4',
                                      'mistral-small-4-119b-2603-nvfp4',
                                      'diffusiongemma-26b-a4b-it-nvfp4'})
        host = {'BIND_HOST': '127.0.0.1', 'PORT': 8000, 'SHUTDOWN_TIMEOUT': 300}
        for profile in loaded.values():
            adapter = lc.ADAPTERS[profile['engine']]
            adapter.validate(profile['native'], profile['metadata'])
            rendered = adapter.render(profile['native'], '/hf-cache/snapshot', host)
            self.assertIn('local-assistant', rendered['resolved'].values())

    def test_laguna_xs_profile_pins_native_context_and_parsers(self):
        profile = lc.profiles.load(lc.ROOT, 'laguna-xs-2.1-nvfp4')
        native = profile['native']
        self.assertEqual(native['model'], 'poolside/Laguna-XS-2.1-NVFP4')
        self.assertEqual(native['revision'], 'd32afde8b09af1539b49ff96ff5551c674485f8e')
        self.assertEqual(native['max-model-len'], 262144)
        self.assertEqual(native['kv-cache-dtype'], 'fp8_e4m3')
        self.assertEqual(native['reasoning-parser'], 'poolside_v1')
        self.assertEqual(native['tool-call-parser'], 'poolside_v1')
        self.assertFalse(profile['metadata']['reasoning-default'])
        self.assertNotIn('speculative-config', native)

    def test_gpt_oss_profile_pins_native_context_and_parsers(self):
        profile = lc.profiles.load(lc.ROOT, 'gpt-oss-120b-mxfp4')
        native = profile['native']
        self.assertEqual(native['model'], 'openai/gpt-oss-120b')
        self.assertEqual(native['revision'], 'b5c939de8f754692c1647ca79fbf85e8c1e70f8a')
        self.assertEqual(native['tokenizer-revision'], native['revision'])
        self.assertEqual(native['max-model-len'], 131072)
        self.assertEqual(native['kv-cache-dtype'], 'fp8_e4m3')
        self.assertEqual(native['reasoning-parser'], 'openai_gptoss')
        self.assertEqual(native['tool-call-parser'], 'openai')
        self.assertTrue(profile['metadata']['reasoning-default'])
        self.assertFalse(profile['metadata']['reasoning-toggle'])
        self.assertTrue(native['enable-prefix-caching'])
        self.assertEqual(profile['metadata']['runtime-environment']['TIKTOKEN_ENCODINGS_BASE'],
                         '/runtime-cache/auxiliary')
        artifact = profile['metadata']['auxiliary-artifacts'][0]
        self.assertEqual(artifact['name'], 'o200k_base.tiktoken')
        self.assertEqual(artifact['sha256'],
                         '446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d')

    def test_muse_glimmer_profile_pins_multimodal_nvfp4_runtime(self):
        profile = lc.profiles.load(lc.ROOT, 'muse-glimmer-30b-nvfp4')
        native = profile['native']
        self.assertEqual(native['model'], 'Inferact/Muse-Glimmer-30B-NVFP4-W4A4')
        self.assertEqual(native['revision'], 'd35cb79050f419c457611b1cee5c5d15b176f285')
        self.assertEqual(native['tokenizer-revision'], native['revision'])
        self.assertEqual(native['max-model-len'], 131072)
        self.assertEqual(native['kv-cache-dtype'], 'auto')
        self.assertEqual(native['reasoning-parser'], 'muse_glimmer')
        self.assertEqual(native['tool-call-parser'], 'muse_glimmer')
        self.assertFalse(profile['metadata']['text-only'])
        self.assertFalse(native['language-model-only'])
        self.assertTrue(profile['metadata']['reasoning-default'])
        self.assertFalse(profile['metadata']['reasoning-toggle'])
        self.assertNotIn('speculative-config', native)

    def test_gemma_4_profile_pins_dgx_spark_multimodal_runtime(self):
        profile = lc.profiles.load(lc.ROOT, 'gemma-4-26b-a4b-nvfp4')
        native = profile['native']
        self.assertEqual(native['model'], 'nvidia/Gemma-4-26B-A4B-NVFP4')
        self.assertEqual(native['revision'], 'a19cfe00be84568a6867111c9a68c9c44fdcffe6')
        self.assertEqual(native['tokenizer-revision'], native['revision'])
        self.assertEqual(native['max-model-len'], 262144)
        self.assertEqual(native['max-num-seqs'], 8)
        self.assertEqual(native['gpu-memory-utilization'], 0.80)
        self.assertEqual(native['max-num-batched-tokens'], 8192)
        self.assertEqual(native['load-format'], 'fastsafetensors')
        self.assertEqual(native['reasoning-parser'], 'gemma4')
        self.assertEqual(native['tool-call-parser'], 'gemma4')
        self.assertFalse(profile['metadata']['text-only'])
        self.assertFalse(native['language-model-only'])
        self.assertFalse(profile['metadata']['reasoning-default'])
        self.assertTrue(profile['metadata']['reasoning-toggle'])
        self.assertEqual(profile['metadata']['runtime-environment']['VLLM_USE_RUST_FRONTEND'], 1)
        self.assertEqual(profile['metadata']['runtime-environment']['VLLM_USE_V2_MODEL_RUNNER'], 1)
        self.assertNotIn('speculative-config', native)
        rendered = lc.runtime_vllm.render(native, '/hf-cache/snapshot',
                                           {'BIND_HOST': '127.0.0.1', 'PORT': 8000,
                                            'SHUTDOWN_TIMEOUT': 300})
        self.assertEqual(rendered['entrypoint'], ['vllm', 'serve'])
        lc.runtime_vllm.inspect_entrypoint(['/opt/nvidia/nvidia_entrypoint.sh'])
        with self.assertRaises(RuntimeError):
            lc.runtime_vllm.inspect_entrypoint(['/bin/sh'])

    def test_mistral_small_4_profile_pins_multimodal_nvfp4_runtime(self):
        profile = lc.profiles.load(lc.ROOT, 'mistral-small-4-119b-2603-nvfp4')
        native = profile['native']
        self.assertEqual(native['model'], 'mistralai/Mistral-Small-4-119B-2603-NVFP4')
        self.assertEqual(native['revision'], '45331841b631f4e281df8e959ea3cc9beb84298a')
        self.assertEqual(native['tokenizer-revision'], native['revision'])
        self.assertEqual(native['max-model-len'], 262144)
        self.assertEqual(native['max-num-seqs'], 1)
        self.assertEqual(native['gpu-memory-utilization'], 0.80)
        self.assertEqual(native['attention-backend'], 'TRITON_MLA')
        self.assertEqual(native['reasoning-parser'], 'mistral')
        self.assertEqual(native['tool-call-parser'], 'mistral')
        self.assertFalse(profile['metadata']['text-only'])
        self.assertFalse(native['language-model-only'])
        self.assertFalse(profile['metadata']['reasoning-default'])
        self.assertTrue(profile['metadata']['reasoning-toggle'])
        self.assertEqual(profile['metadata']['reasoning-control'], 'reasoning-effort')
        self.assertNotIn('speculative-config', native)

    def test_diffusiongemma_profile_pins_dgx_spark_diffusion_runtime(self):
        profile = lc.profiles.load(lc.ROOT, 'diffusiongemma-26b-a4b-it-nvfp4')
        native = profile['native']
        self.assertEqual(native['model'], 'nvidia/diffusiongemma-26B-A4B-it-NVFP4')
        self.assertEqual(native['revision'], 'ec4ff3df205028f4e81c954c2227f9312b3ec2ea')
        self.assertEqual(native['tokenizer-revision'], native['revision'])
        self.assertEqual(native['max-model-len'], 262144)
        self.assertEqual(native['max-num-seqs'], 8)
        self.assertEqual(native['gpu-memory-utilization'], 0.80)
        self.assertEqual(native['load-format'], 'fastsafetensors')
        self.assertEqual(native['attention-backend'], 'TRITON_ATTN')
        self.assertEqual(native['diffusion-config'], {'canvas_length': 256})
        self.assertIsNone(native['override-generation-config']['max_new_tokens'])
        self.assertEqual(native['reasoning-parser'], 'gemma4')
        self.assertEqual(native['tool-call-parser'], 'gemma4')
        self.assertFalse(profile['metadata']['text-only'])
        self.assertFalse(native['language-model-only'])
        self.assertTrue(profile['metadata']['reasoning-default'])
        self.assertTrue(profile['metadata']['reasoning-toggle'])

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

    def test_auxiliary_artifact_is_hash_verified_from_runtime_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'auxiliary' / 'vocab.tiktoken'
            target.parent.mkdir()
            target.write_bytes(b'pinned-vocab')
            sha256 = hashlib.sha256(b'pinned-vocab').hexdigest()
            metadata = {'auxiliary-artifacts': [{
                'name': 'vocab.tiktoken', 'url': 'https://invalid.example/vocab',
                'sha256': sha256}]}
            lc.ensure_auxiliary_artifacts(root, metadata)
            rec = {'compose_env': {'RUNTIME_CACHE_DIR': str(root)},
                   'identity': {'profile_policy': metadata}}
            lc.verify_auxiliary_artifacts(rec)
            target.write_bytes(b'tampered')
            with self.assertRaisesRegex(RuntimeError, 'missing or invalid'):
                lc.verify_auxiliary_artifacts(rec)

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
