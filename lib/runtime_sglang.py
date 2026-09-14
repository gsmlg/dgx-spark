"""SGLang-specific validation, probing, and release rendering."""
import json
import re

import yaml


MODEL_KEYS = {'model-path', 'revision', 'tokenizer-revision', 'served-model-name', 'tp-size',
              'context-length', 'max-running-requests', 'mem-fraction-static',
              'chunked-prefill-size', 'ple-offload-embedding', 'ple-offload-backend',
              'ple-offload-dir', 'moe-runner-backend', 'fp4-gemm-backend', 'page-size',
              'max-mamba-cache-size', 'reasoning-parser', 'tool-call-parser', 'trust-remote-code'}


def validate(native, metadata):
    if set(native) != MODEL_KEYS:
        raise RuntimeError('sglang.yaml has missing or unknown settings')
    if not re.fullmatch(r'[0-9a-f]{40}', str(native['revision'])) or native['tokenizer-revision'] != native['revision']:
        raise RuntimeError('Model and tokenizer revisions must be the same full commit')
    for key in ('tp-size', 'context-length', 'max-running-requests', 'chunked-prefill-size',
                'page-size', 'max-mamba-cache-size'):
        if type(native[key]) is not int or native[key] <= 0:
            raise RuntimeError(f'{key} must be a positive integer')
    if type(native['mem-fraction-static']) not in (int, float):
        raise RuntimeError('mem-fraction-static must be numeric')
    if native['tp-size'] != 1 or not 0 < native['mem-fraction-static'] < 1:
        raise RuntimeError('Invalid SGLang memory fraction or tensor parallelism')
    if native['context-length'] != metadata['context-tokens'] or native['max-running-requests'] != metadata['max-running-requests']:
        raise RuntimeError('Native context/concurrency does not match profile policy')
    if native['ple-offload-embedding'] is not True or native['ple-offload-backend'] != 'file' or native['ple-offload-dir'] != '/ple-cache':
        raise RuntimeError('Flash-Next requires the owned file-backed PLE cache')
    if native['moe-runner-backend'] != 'flashinfer_cutlass' or native['fp4-gemm-backend'] != 'flashinfer_cutlass':
        raise RuntimeError('Flash-Next requires the validated flashinfer_cutlass FP4 and MoE backends')
    if native['trust-remote-code'] is not True:
        raise RuntimeError('The pinned Flash-Next implementation requires trust-remote-code')


def repository(native):
    return native['model-path'], native['revision']


def inspect_entrypoint(entrypoint):
    # The frozen release overrides the image entrypoint explicitly.
    if entrypoint is not None and not isinstance(entrypoint, list):
        raise RuntimeError('Candidate SGLang entrypoint metadata is malformed')


def cuda_probe():
    return ('import torch,sglang,json; x=torch.ones((32,32),device="cuda"); y=x@x; '
            'torch.cuda.synchronize(); assert y[0,0].item()==32; '
            'print(json.dumps({"torch":torch.__version__,"engine":getattr(sglang,"__version__","unknown"),'
            '"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),'
            '"capability":torch.cuda.get_device_capability()}))')


def probe_command(pinned):
    return ['docker', 'run', '--rm', '--gpus', 'all', '--entrypoint', 'python3', pinned,
            '-m', 'sglang.launch_server', '--help']


def validate_help(help_text, native, metadata):
    for key in set(native) - {'revision', 'tokenizer-revision'}:
        if '--' + key not in help_text:
            raise RuntimeError(f'Candidate SGLang runtime does not support --{key}')
    for feature in metadata['required-runtime-features']:
        if feature not in help_text:
            raise RuntimeError(f'Candidate SGLang runtime does not provide {feature}')
    if '--api-key' not in help_text:
        raise RuntimeError('Candidate SGLang runtime does not support API authentication')


def render(native, snapshot, host, authenticated=False):
    resolved = dict(native)
    resolved['model-path'] = snapshot
    resolved.update({'host': host['BIND_HOST'], 'port': host['PORT']})
    command = []
    for key, value in resolved.items():
        if key in ('revision', 'tokenizer-revision'):
            continue
        flag = '--' + key
        if value is True:
            command.append(flag)
        elif value is not False and value is not None:
            command.extend([flag, str(value)])
    if authenticated:
        command.extend(['--api-key', '${VLLM_API_KEY:?}'])
    return {'config_name': 'sglang.yaml', 'config_text': yaml.safe_dump(resolved, sort_keys=False),
            'entrypoint': ['python3', '-m', 'sglang.launch_server'], 'command': command,
            'resolved': resolved, 'environment': {'VLLM_API_KEY': '${VLLM_API_KEY:-}'}}
