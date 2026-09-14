"""vLLM-specific validation, probing, and release rendering."""
import json
import re

import yaml


MODEL_KEYS = {'model', 'revision', 'tokenizer-revision', 'served-model-name',
              'tensor-parallel-size', 'max-model-len', 'max-num-seqs', 'gpu-memory-utilization',
              'max-num-batched-tokens', 'enable-chunked-prefill', 'enable-prefix-caching',
              'dtype', 'kv-cache-dtype', 'language-model-only', 'reasoning-parser',
              'tool-call-parser', 'enable-auto-tool-choice', 'default-chat-template-kwargs',
              'generation-config', 'override-generation-config', 'enable-log-requests',
              'enable-log-outputs', 'disable-uvicorn-access-log', 'enforce-eager'}
OPTIONAL_KEYS = {'enable-flashinfer-autotune'}
OPTIONAL_KEYS = {'enforce-eager'}
BOOL_KEYS = {'enable-chunked-prefill', 'enable-prefix-caching', 'language-model-only',
             'enable-auto-tool-choice', 'enable-log-requests', 'enable-log-outputs',
             'disable-uvicorn-access-log', 'enforce-eager'}


def validate(native, metadata):
    if set(native) - OPTIONAL_KEYS != MODEL_KEYS - OPTIONAL_KEYS or set(native) - MODEL_KEYS:
        raise RuntimeError('vllm.yaml has missing or unknown settings')
    if not re.fullmatch(r'[0-9a-f]{40}', str(native['revision'])):
        raise RuntimeError('Model revision must be a full 40-character commit')
    if native['tokenizer-revision'] != native['revision']:
        raise RuntimeError('Tokenizer and checkpoint revisions must match')
    for key in ('tensor-parallel-size', 'max-model-len', 'max-num-seqs', 'max-num-batched-tokens'):
        if type(native[key]) is not int or native[key] <= 0:
            raise RuntimeError(f'{key} must be a positive integer')
    if any(type(native[key]) is not bool for key in BOOL_KEYS if key in native):
        raise RuntimeError('vLLM boolean settings must be YAML booleans')
    if type(native['gpu-memory-utilization']) not in (int, float):
        raise RuntimeError('gpu-memory-utilization must be numeric')
    if native['tensor-parallel-size'] != 1 or not 0 < native['gpu-memory-utilization'] < 1:
        raise RuntimeError('Invalid GPU allocation or tensor parallelism')
    if native['max-model-len'] != metadata['context-tokens'] or native['max-num-seqs'] != metadata['max-running-requests']:
        raise RuntimeError('Native context/concurrency does not match profile policy')
    if not native['language-model-only'] or native['enable-log-requests'] or native['enable-log-outputs']:
        raise RuntimeError('Text-only mode and disabled prompt/output logs are required')
    if not isinstance(native['override-generation-config'], dict) or 'max_new_tokens' in native['override-generation-config']:
        raise RuntimeError('Invalid generation override configuration')


def repository(native):
    return native['model'], native['revision']


def inspect_entrypoint(entrypoint):
    if entrypoint != ['vllm', 'serve']:
        raise RuntimeError(f'Unsupported vLLM image entrypoint {entrypoint}')


def cuda_probe():
    return ('import torch,vllm,json; x=torch.ones((32,32),device="cuda"); y=x@x; '
            'torch.cuda.synchronize(); assert y[0,0].item()==32; '
            'print(json.dumps({"torch":torch.__version__,"engine":vllm.__version__,"cuda":torch.version.cuda,'
            '"gpu":torch.cuda.get_device_name(),"capability":torch.cuda.get_device_capability()}))')


def probe_command(pinned):
    return ['docker', 'run', '--rm', '--gpus', 'all', pinned, '--help=all']


def validate_help(help_text, native, metadata):
    for key in native.keys() | {'host', 'port', 'shutdown-timeout'}:
        if '--' + key not in help_text:
            raise RuntimeError(f'Candidate vLLM runtime does not support --{key}')
    for feature in metadata['required-runtime-features']:
        if feature not in help_text:
            raise RuntimeError(f'Candidate vLLM runtime does not provide {feature}')


def render(native, snapshot, host, authenticated=False):
    resolved = dict(native)
    resolved.update({'model': snapshot, 'tokenizer': snapshot, 'host': host['BIND_HOST'],
                     'port': host['PORT'], 'shutdown-timeout': host['SHUTDOWN_TIMEOUT']})
    return {'config_name': 'vllm.yaml', 'config_text': yaml.safe_dump(resolved, sort_keys=False),
            'entrypoint': None, 'command': ['--config', '/release/vllm.yaml'], 'resolved': resolved,
            'environment': {'VLLM_API_KEY': '${VLLM_API_KEY:-}', 'VLLM_CACHE_ROOT': '/runtime-cache/vllm'}}
