"""Validate complete model snapshot layouts without assuming Transformers filenames."""
import json


TRANSFORMERS_PRIMARY = {
    'config.json', 'tokenizer_config.json', 'tokenizer.json',
    'model.safetensors.index.json',
}
MISTRAL_NATIVE_PRIMARY = {
    'params.json', 'tokenizer_config.json', 'tokenizer.json', 'tekken.json',
    'chat_template.jinja', 'consolidated.safetensors.index.json',
}
WEIGHT_INDEXES = ('model.safetensors.index.json',
                  'consolidated.safetensors.index.json')


def validate_artifact_layout(root, paths, kind):
    """Reject incomplete primary snapshots and dangling sharded-weight indexes."""
    if kind == 'primary' and not (
            TRANSFORMERS_PRIMARY.issubset(paths)
            or MISTRAL_NATIVE_PRIMARY.issubset(paths)):
        raise RuntimeError('Required model/tokenizer artifacts missing')
    if not ({'config.json', 'params.json'} & paths) or not any(
            path.endswith('.safetensors') for path in paths):
        raise RuntimeError('Required model weights or configuration missing')
    for name in WEIGHT_INDEXES:
        index_path = root / name
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text())
        weight_map = index.get('weight_map') if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f'Invalid weight index: {name}')
        if not all((root / shard).is_file() for shard in set(weight_map.values())):
            raise RuntimeError('Weight index references missing shards')
