"""Strict, side-effect-free model profile loading."""
from pathlib import Path
import re

import yaml


DEFAULT_PROFILE = 'qwen38-27b-nvfp4'
PROFILE_KEYS = {
    'schema-version', 'id', 'engine', 'display-name', 'text-only',
    'reasoning-default', 'reasoning-toggle', 'reasoning-fields', 'tool-support',
    'context-tokens', 'max-running-requests', 'runtime-environment',
    'required-runtime-features', 'derived-cache',
}
OPTIONAL_PROFILE_KEYS = {'auxiliary-artifacts', 'reasoning-control'}
AUXILIARY_ARTIFACT_KEYS = {'name', 'url', 'sha256'}
DERIVED_CACHE_KEYS = {'kind', 'mount', 'minimum-free-gib', 'reset-before-start'}
ENGINES = {'vllm': 'vllm.yaml', 'sglang': 'sglang.yaml'}
RUNTIME_ENV_KEYS = {'MAX_JOBS', 'CUTE_DSL_ARCH', 'PYTORCH_CUDA_ALLOC_CONF',
                    'SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB', 'TIKTOKEN_ENCODINGS_BASE',
                    'VLLM_USE_RUST_FRONTEND', 'VLLM_USE_V2_MODEL_RUNNER'}


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise RuntimeError(f'Duplicate YAML setting: {key}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _yaml(path):
    if not path.is_file():
        raise RuntimeError(f'Missing profile file: {path.name}')
    raw = path.read_text()
    if '$' in raw:
        raise RuntimeError(f'Unresolved variable in {path.name}')
    value = yaml.load(raw, Loader=UniqueLoader)
    if not isinstance(value, dict):
        raise RuntimeError(f'{path.name} must contain a mapping')
    return value


def resolve(root, profile_id=DEFAULT_PROFILE):
    if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,63}', profile_id or ''):
        raise RuntimeError('Invalid profile ID')
    profiles = (root / 'profiles').resolve()
    candidate = profiles / profile_id
    path = candidate.resolve()
    if path.parent != profiles or not path.is_dir() or candidate.is_symlink():
        raise RuntimeError(f'Unknown profile: {profile_id}')
    return path


def load(root, profile_id=DEFAULT_PROFILE):
    path = resolve(root, profile_id)
    metadata = _yaml(path / 'profile.yaml')
    if set(metadata) - OPTIONAL_PROFILE_KEYS != PROFILE_KEYS or set(metadata) - PROFILE_KEYS - OPTIONAL_PROFILE_KEYS:
        raise RuntimeError('profile.yaml has missing or unknown settings')
    if metadata['schema-version'] != 1 or metadata['id'] != profile_id:
        raise RuntimeError('Profile schema version or ID does not match its directory')
    engine = metadata['engine']
    if engine not in ENGINES:
        raise RuntimeError(f'Unknown engine: {engine}')
    for key in ('text-only', 'reasoning-default', 'reasoning-toggle', 'tool-support'):
        if type(metadata[key]) is not bool:
            raise RuntimeError(f'{key} must be a YAML boolean')
    reasoning_control = metadata.get('reasoning-control', 'enable-thinking')
    if reasoning_control not in ('enable-thinking', 'reasoning-effort'):
        raise RuntimeError('Unknown reasoning control')
    if reasoning_control == 'reasoning-effort' and not metadata['reasoning-toggle']:
        raise RuntimeError('reasoning-effort control requires a toggleable profile')

    for key in ('context-tokens', 'max-running-requests'):
        if type(metadata[key]) is not int or metadata[key] <= 0:
            raise RuntimeError(f'{key} must be a positive integer')
    if not isinstance(metadata['reasoning-fields'], list) or not metadata['reasoning-fields']:
        raise RuntimeError('reasoning-fields must be a non-empty list')
    if not all(re.fullmatch(r'[a-z_][a-z0-9_]*', x or '') for x in metadata['reasoning-fields']):
        raise RuntimeError('Invalid reasoning response field')
    runtime_environment = metadata['runtime-environment']
    if not isinstance(runtime_environment, dict) or not all(
            re.fullmatch(r'[A-Z][A-Z0-9_]*', str(k)) and type(v) in (str, int, float)
            for k, v in runtime_environment.items()) or not set(runtime_environment) <= RUNTIME_ENV_KEYS:
        raise RuntimeError('runtime-environment must contain literal allowlisted values')
    features = metadata['required-runtime-features']
    if not isinstance(features, list) or not all(isinstance(x, str) and x for x in features):
        raise RuntimeError('required-runtime-features must be a list of strings')
    auxiliary = metadata.get('auxiliary-artifacts', [])
    if not isinstance(auxiliary, list):
        raise RuntimeError('auxiliary-artifacts must be a list')
    for artifact in auxiliary:
        if not isinstance(artifact, dict) or set(artifact) != AUXILIARY_ARTIFACT_KEYS:
            raise RuntimeError('Invalid auxiliary artifact')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', artifact['name'] or ''):
            raise RuntimeError('Invalid auxiliary artifact name')
        if not re.fullmatch(r'https://[^\s]+', artifact['url'] or ''):
            raise RuntimeError('Auxiliary artifact URL must use HTTPS')
        if not re.fullmatch(r'[0-9a-f]{64}', artifact['sha256'] or ''):
            raise RuntimeError('Auxiliary artifact SHA256 must be lowercase hexadecimal')
    derived = metadata['derived-cache']
    if derived is not None:
        if not isinstance(derived, dict) or set(derived) != DERIVED_CACHE_KEYS:
            raise RuntimeError('Invalid derived-cache policy')
        if derived['kind'] != 'ple' or derived['mount'] != '/ple-cache':
            raise RuntimeError('Only the owned /ple-cache derived cache is supported')
        if type(derived['minimum-free-gib']) is not int or derived['minimum-free-gib'] <= 0:
            raise RuntimeError('derived-cache minimum-free-gib must be positive')
        if derived['reset-before-start'] is not True:
            raise RuntimeError('PLE derived cache must use the safe fresh-file startup policy')
    native = _yaml(path / ENGINES[engine])
    image_path = path / 'image.env'
    return {'path': path, 'metadata': metadata, 'native': native,
            'image_path': image_path, 'engine': engine}


def list_profiles(root):
    result = []
    for path in sorted((root / 'profiles').iterdir()):
        if path.is_dir() and not path.is_symlink():
            try:
                result.append(load(root, path.name)['metadata'])
            except RuntimeError:
                continue
    return result
