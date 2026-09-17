"""vLLM-specific validation, probing, and release rendering."""
import json
from pathlib import Path
import re

import yaml


MODEL_KEYS = {'model', 'revision', 'tokenizer-revision', 'served-model-name',
              'tensor-parallel-size', 'max-model-len', 'max-num-seqs', 'gpu-memory-utilization',
              'max-num-batched-tokens', 'enable-chunked-prefill', 'enable-prefix-caching',
              'dtype', 'kv-cache-dtype', 'tokenizer-mode', 'language-model-only', 'reasoning-parser',
              'tool-call-parser', 'enable-auto-tool-choice', 'default-chat-template-kwargs',
              'generation-config', 'override-generation-config', 'enable-log-requests',
              'enable-log-outputs', 'disable-uvicorn-access-log', 'enforce-eager',
              'enable-flashinfer-autotune', 'load-format', 'attention-backend',
              'diffusion-config', 'speculative-config', 'middleware'}
OPTIONAL_KEYS = {'enable-flashinfer-autotune', 'enforce-eager', 'load-format', 'tokenizer-mode',
                 'attention-backend', 'diffusion-config', 'speculative-config', 'middleware'}
BOOL_KEYS = {'enable-chunked-prefill', 'enable-prefix-caching', 'language-model-only',
             'enable-auto-tool-choice', 'enable-log-requests', 'enable-log-outputs',
             'disable-uvicorn-access-log', 'enforce-eager'}
RESPONSE_MIDDLEWARE = 'vllm_response_compat.ResponsesMessageMiddleware'


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
    if native['language-model-only'] != metadata['text-only']:
        raise RuntimeError('Native language-only mode must match the profile input policy')
    if native['enable-log-requests'] or native['enable-log-outputs']:
        raise RuntimeError('Prompt and output logs must be disabled')
    if native.get('load-format') not in (None, 'fastsafetensors'):
        raise RuntimeError('Unsupported vLLM load format')
    if 'tokenizer-mode' in native and native['tokenizer-mode'] not in ('auto', 'hf', 'slow', 'mistral'):
        raise RuntimeError('Unsupported tokenizer mode')
    if 'middleware' in native and native['middleware'] != [RESPONSE_MIDDLEWARE]:
        raise RuntimeError('Unsupported vLLM middleware')
    if native.get('attention-backend') not in (None, 'TRITON_ATTN', 'TRITON_MLA'):
        raise RuntimeError('Unsupported attention backend')
    if native.get('diffusion-config') not in (None, {'canvas_length': 256}):
        raise RuntimeError('Unsupported diffusion configuration')
    speculative = native.get('speculative-config')
    if speculative is not None:
        if not isinstance(speculative, dict) or set(speculative) != {
                'method', 'model', 'num_speculative_tokens'}:
            raise RuntimeError('Invalid speculative decoding configuration')
        if speculative['method'] != 'dflash' or type(speculative['num_speculative_tokens']) is not int \
                or speculative['num_speculative_tokens'] <= 0:
            raise RuntimeError('Unsupported speculative decoding configuration')
        names = {model['name'] for model in metadata.get('auxiliary-models', [])}
        if speculative['model'] not in names:
            raise RuntimeError('Speculative model must reference a pinned auxiliary model')
    override = native['override-generation-config']
    if not isinstance(override, dict) or (
            'max_new_tokens' in override and override['max_new_tokens'] is not None):
        raise RuntimeError('Invalid generation override configuration')


def repository(native):
    return native['model'], native['revision']


def inspect_entrypoint(entrypoint):
    if entrypoint not in (['vllm', 'serve'], ['/opt/nvidia/nvidia_entrypoint.sh']):
        raise RuntimeError(f'Unsupported vLLM image entrypoint {entrypoint}')


def cuda_probe():
    return ('import torch,vllm,json; x=torch.ones((32,32),device="cuda"); y=x@x; '
            'torch.cuda.synchronize(); assert y[0,0].item()==32; '
            'print(json.dumps({"torch":torch.__version__,"engine":vllm.__version__,"cuda":torch.version.cuda,'
            '"gpu":torch.cuda.get_device_name(),"capability":torch.cuda.get_device_capability()}))')


def probe_command(pinned):
    code = (
        "import subprocess; "
        "result=subprocess.run(['vllm','serve','--help=all'],check=True,text=True,"
        "stdout=subprocess.PIPE); print(result.stdout); "
        "from vllm.model_executor.model_loader import LoadFormats; "
        "print('\\n'.join(LoadFormats.__args__)); "
        "from vllm.v1.attention.backends.registry import AttentionBackendEnum; "
        "print('\\n'.join(item.name for item in AttentionBackendEnum))"
    )
    return ['docker', 'run', '--rm', '--gpus', 'all', '--entrypoint', 'python3',
            pinned, '-c', code]


def validate_help(help_text, native, metadata):
    for key in native.keys() | {'host', 'port', 'shutdown-timeout'}:
        if '--' + key not in help_text:
            raise RuntimeError(f'Candidate vLLM runtime does not support --{key}')
    for feature in metadata['required-runtime-features']:
        if feature not in help_text:
            raise RuntimeError(f'Candidate vLLM runtime does not provide {feature}')


def render(native, snapshot, host, authenticated=False, auxiliary_models=None,
           rust_frontend=False):
    resolved = dict(native)
    files = ({'vllm_response_compat.py': Path(__file__).with_name('vllm_response_compat.py').read_text()}
             if RESPONSE_MIDDLEWARE in native.get('middleware', []) else {})
    if 'speculative-config' in resolved:
        speculative = dict(resolved['speculative-config'])
        try:
            speculative['model'] = (auxiliary_models or {})[speculative['model']]
        except KeyError as error:
            raise RuntimeError('Pinned speculative model snapshot is unavailable') from error
        resolved['speculative-config'] = speculative
    resolved.update({'model': snapshot, 'tokenizer': snapshot, 'host': host['BIND_HOST'],
                     'port': host['PORT'], 'shutdown-timeout': host['SHUTDOWN_TIMEOUT']})
    command = ['--config', '/release/vllm.yaml']
    if rust_frontend:
        # vllm-rs 0.28 requires MODEL immediately after `serve` and does not
        # implement --config. Unknown engine flags are forwarded to Python.
        command = [snapshot]
        for key, value in resolved.items():
            if key in ('model', 'tokenizer', 'tokenizer-revision') or value is False or value is None:
                continue
            flag = '--' + key
            if value is True:
                command.append(flag)
            else:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, separators=(',', ':'))
                command.extend([flag, str(value)])
    environment = {'VLLM_API_KEY': '${VLLM_API_KEY:-}', 'VLLM_CACHE_ROOT': '/runtime-cache/vllm'}
    if files:
        environment['PYTHONPATH'] = '/release'
    return {'config_name': 'vllm.yaml', 'config_text': yaml.safe_dump(resolved, sort_keys=False),
            'entrypoint': ['vllm', 'serve'], 'command': command,
            'resolved': resolved, 'files': files, 'environment': environment}
