"""Run inside the candidate image, without GPU access or a resident service."""
import hashlib
import json
import os
from pathlib import Path
import sys

from huggingface_hub import HfApi, snapshot_download

repo, revision = sys.argv[1:3]
api = HfApi(token=os.environ.get('HF_TOKEN') or False)
info = api.model_info(repo, revision=revision, files_metadata=True)
if info.sha != revision:
    raise RuntimeError('Hub revision does not match requested full commit')
root = Path(snapshot_download(repo, revision=revision, cache_dir='/hf-cache',
                              token=os.environ.get('HF_TOKEN') or False, max_workers=4))
manifest = []
for entry in info.siblings:
    path = root / entry.rfilename
    if not path.is_file() or path.stat().st_size != entry.size:
        raise RuntimeError(f'Incomplete artifact: {entry.rfilename}')
    sha = hashlib.sha256()
    git = hashlib.sha1(f'blob {entry.size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
            git.update(chunk)
    if entry.lfs:
        if sha.hexdigest() != entry.lfs.sha256:
            raise RuntimeError(f'SHA256 mismatch: {entry.rfilename}')
    elif git.hexdigest() != entry.blob_id:
        raise RuntimeError(f'Git blob mismatch: {entry.rfilename}')
    manifest.append({'path': entry.rfilename, 'size': entry.size, 'sha256': sha.hexdigest()})
required = {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors.index.json'}
if not required.issubset({x['path'] for x in manifest}):
    raise RuntimeError('Required model/tokenizer artifacts missing')
index = json.loads((root / 'model.safetensors.index.json').read_text())
if not all((root / name).is_file() for name in set(index['weight_map'].values())):
    raise RuntimeError('Weight index references missing shards')
print('SPARK_MANIFEST=' + json.dumps({'snapshot': str(root), 'files': manifest}))
