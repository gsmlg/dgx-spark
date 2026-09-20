#!/usr/bin/env python3
"""Small host-side Docker Compose lifecycle, with side-effect-free config checks."""
import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

try:
    import yaml
except ImportError:
    sys.exit('Python PyYAML is required on the host (python3-yaml).')

import profiles
import runtime_sglang
import runtime_vllm

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'state'
HOST_KEYS = {'HF_CACHE', 'RUNTIME_CACHE', 'BIND_HOST', 'PORT', 'STARTUP_TIMEOUT',
             'REQUEST_TIMEOUT', 'HEALTH_TIMEOUT', 'SHUTDOWN_TIMEOUT', 'STOP_GRACE_SECONDS',
             'DISK_RESERVE_GIB', 'PREPARE_OVERHEAD_GIB', 'MIN_AVAILABLE_GIB'}
MEMORY_GUARD_GRACE_SECONDS = 15
ADAPTERS = {'vllm': runtime_vllm, 'sglang': runtime_sglang}


def fail(message):
    raise RuntimeError(message)


def run(args, *, env=None, capture=True, timeout=None, check=True):
    result = subprocess.run([str(a) for a in args], env=env, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None, timeout=timeout)
    if check and result.returncode:
        detail = (result.stderr or '')[-6000:]
        for source in (env or {}, env_file(ROOT / 'secrets.env', {'HF_TOKEN', 'VLLM_API_KEY'}, False)):
            for key, value in source.items():
                if value and any(part in key.upper() for part in ('TOKEN', 'KEY', 'SECRET', 'PASSWORD', 'PROXY')):
                    detail = detail.replace(str(value), '[REDACTED]')
        fail(f'{args[0]} operation failed (exit {result.returncode}). '
             + detail)
    return result.stdout.strip() if capture else result.returncode


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.tmp-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def env_file(path, allowed, required=True):
    if not path.exists():
        if required:
            fail(f'Missing {path.name}; copy the example first.')
        return {}
    values = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        if not sep or key not in allowed or key in values:
            fail(f'{path.name}:{n}: unknown, duplicate or invalid key')
        if any(c in value for c in ('$', '`', '\n', '\r', '"', "'")):
            fail(f'{path.name}:{n}: use literal unquoted values, without substitutions')
        values[key] = value
    return values


UniqueLoader = profiles.UniqueLoader


def host_config(require_secrets=True):
    host = env_file(ROOT / 'host.env', HOST_KEYS)
    if set(host) != HOST_KEYS:
        fail('host.env must contain all documented host keys')
    for k in HOST_KEYS - {'HF_CACHE', 'RUNTIME_CACHE', 'BIND_HOST'}:
        try:
            host[k] = int(host[k])
        except ValueError:
            fail(f'{k} must be an integer')
        if host[k] <= 0:
            fail(f'{k} must be positive')
    if host['PORT'] > 65535 or host['STOP_GRACE_SECONDS'] <= host['SHUTDOWN_TIMEOUT']:
        fail('Invalid port or stop grace period (must exceed shutdown timeout)')
    for key in ('HF_CACHE', 'RUNTIME_CACHE'):
        if not Path(host[key]).is_absolute() or any(x in host[key] for x in (',', '\n')):
            fail(f'{key} must be an absolute path without commas/newlines')
    try:
        address = ipaddress.ip_address(host['BIND_HOST'])
    except ValueError:
        fail('BIND_HOST must be an explicit IP address')
    if not address.is_loopback and (not address.is_private or address.is_unspecified):
        fail('Bind to loopback or a specific private interface address')
    secrets = env_file(ROOT / 'secrets.env', {'HF_TOKEN', 'VLLM_API_KEY'}, False) if require_secrets else {}
    if (ROOT / 'secrets.env').exists() and (ROOT / 'secrets.env').stat().st_mode & 0o077:
        fail('Run chmod 600 secrets.env before using credentials')
    if not address.is_loopback and not secrets.get('VLLM_API_KEY'):
        fail('A private-network bind requires VLLM_API_KEY in secrets.env')
    return host, secrets


def secrets_config():
    secrets = env_file(ROOT / 'secrets.env', {'HF_TOKEN', 'VLLM_API_KEY'}, False)
    if (ROOT / 'secrets.env').exists() and (ROOT / 'secrets.env').stat().st_mode & 0o077:
        fail('Run chmod 600 secrets.env before using credentials')
    return secrets


def config(profile_id=profiles.DEFAULT_PROFILE):
    host, secrets = host_config()
    profile = profiles.load(ROOT, profile_id)
    ADAPTERS[profile['engine']].validate(profile['native'], profile['metadata'])
    image = env_file(profile['image_path'], {'IMAGE'})
    if set(image) != {'IMAGE'} or not re.fullmatch(r'[a-zA-Z0-9._/@:-]+', image['IMAGE']):
        fail('image.env needs one literal IMAGE reference')
    return host, secrets, profile, image['IMAGE']


def containers():
    ids = run(['docker', 'ps', '-aq', '--filter', 'label=com.docker.compose.project=spark-llm']).split()
    return json.loads(run(['docker', 'inspect', *ids])) if ids else []


def conflicts():
    ours = {c['Id'] for c in containers()}
    ids = run(['docker', 'ps', '-q']).split()
    rows = json.loads(run(['docker', 'inspect', *ids])) if ids else []
    return [c['Name'].lstrip('/') for c in rows if c['Id'] not in ours and
            (c['HostConfig'].get('DeviceRequests') or c['HostConfig'].get('Runtime') == 'nvidia' or
             any(x in str(c['Config'].get('Cmd', [])).lower() for x in ('vllm', 'ollama', 'sglang')))]


def memory_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    fail('Cannot read host MemAvailable')


def require_idle(host):
    busy = conflicts()
    if busy:
        fail('Conflicting GPU containers: ' + ', '.join(busy) + '. Stop them explicitly before launch.')
    processes = run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    if processes.strip():
        fail('GPU compute processes are still active; wait for resource release or stop their owner explicitly.')
    if memory_available() < host['MIN_AVAILABLE_GIB'] * 2**30:
        fail('Host available memory is below the configured guardrail')
    family = socket.AF_INET6 if ':' in host['BIND_HOST'] else socket.AF_INET
    with socket.socket(family) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host['BIND_HOST'], host['PORT']))
        except OSError:
            fail('Configured API address/port is already in use')


def doctor(host):
    if platform.machine() not in ('aarch64', 'arm64'):
        fail('The target must be native ARM64')
    report = {'time': time.time(), 'os': platform.platform(), 'architecture': platform.machine(),
              'docker': run(['docker', 'version', '--format', '{{.Server.Version}}']),
              'compose': run(['docker', 'compose', 'version', '--short']),
              'gpu': run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader']),
              'toolkit': run(['nvidia-container-cli', '--version']),
              'mem_available_bytes': memory_available(), 'conflicting_containers': conflicts(),
              'workspace_free_bytes': shutil.disk_usage(ROOT).free,
              'image_cuda_validation': 'pending; performed by prepare'}
    atomic(ROOT / 'reports/doctor.json', report)
    print(json.dumps(report, indent=2))
    return report


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def release_path(release_id):
    if not re.fullmatch(r'[0-9a-f]{24}', release_id or ''):
        fail('Invalid release ID')
    path = STATE / 'releases' / release_id
    if not path.is_dir():
        fail('Release does not exist; run prepare')
    return path


def release(release_id):
    path = release_path(release_id)
    rec = read_json(path / 'release.json')
    if rec is None or digest_object(rec['identity'])[:24] != release_id:
        fail('Release identity mismatch')
    for name, expected in rec['hashes'].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            fail(f'Release was modified: {name}; prepare a new release')
    return path, rec


def compose(release_id, secrets, *args, capture=True):
    path, rec = release(release_id)
    values = rec['compose_env'] | {'RELEASE_DIR': str(path),
                                  'VLLM_API_KEY': secrets.get('VLLM_API_KEY', '')}
    # Do not inherit ambient Compose/VLLM/HF overrides.
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('COMPOSE_', 'VLLM_', 'HF_'))}
    environment.update({k: str(v) for k, v in values.items()})
    return run(['docker', 'compose', '--project-name', 'spark-llm', '--env-file', '/dev/null',
                '-f', path / 'compose.yaml', *args], env=environment, capture=capture)


def render_compose(runtime, metadata):
    service = {
        'image': '${IMAGE:?Use bin/spark-llm prepare first}', 'platform': 'linux/arm64',
        'pull_policy': 'never', 'command': runtime['command'], 'network_mode': 'host',
        'ipc': 'host', 'restart': 'no', 'stop_grace_period': '${STOP_GRACE_SECONDS:?}s',
        'environment': {
            'HF_HUB_CACHE': '/hf-cache', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
            'XDG_CACHE_HOME': '/runtime-cache', 'SPARK_HEALTH_HOST': '${HEALTH_HOST:?}',
            'SPARK_HEALTH_PORT': '${PORT:?}', 'SPARK_HEALTH_TIMEOUT': '${HEALTH_TIMEOUT:?}',
            **{k: str(v) for k, v in metadata['runtime-environment'].items()},
            **runtime['environment'],
        },
        'volumes': [
            {'type': 'bind', 'source': '${HF_CACHE:?}', 'target': '/hf-cache', 'read_only': True},
            {'type': 'bind', 'source': '${RELEASE_DIR:?}', 'target': '/release', 'read_only': True},
            {'type': 'bind', 'source': '${RUNTIME_CACHE_DIR:?}', 'target': '/runtime-cache'},
        ],
        'deploy': {'resources': {'reservations': {'devices': [
            {'driver': 'nvidia', 'count': 1, 'capabilities': ['gpu']} ]}}},
        'healthcheck': {
            'test': ['CMD', 'python3', '-c',
                     "import os,urllib.request;h=os.environ['SPARK_HEALTH_HOST'];"
                     "h='['+h+']' if ':' in h else h;u='http://'+h+':'+os.environ['SPARK_HEALTH_PORT']+'/health';"
                     "r=urllib.request.Request(u,headers={'Authorization':'Bearer '+os.environ.get('VLLM_API_KEY','')});"
                     "urllib.request.urlopen(r,timeout=float(os.environ['SPARK_HEALTH_TIMEOUT'])).close()"],
            'interval': '15s', 'timeout': '${HEALTH_TIMEOUT:?}s', 'retries': 4,
            'start_period': '${STARTUP_TIMEOUT:?}s'},
        'logging': {'driver': 'json-file', 'options': {'max-size': '20m', 'max-file': '3'}},
    }
    if runtime['entrypoint']:
        service['entrypoint'] = runtime['entrypoint']
    if metadata['derived-cache']:
        service['volumes'].append({'type': 'bind', 'source': '${PLE_CACHE_DIR:?}',
                                   'target': metadata['derived-cache']['mount']})
    return yaml.safe_dump({'name': 'spark-llm', 'services': {'llm': service}}, sort_keys=False)


def check_disk_capacity(host, metadata):
    for target in (Path(host['HF_CACHE']), Path(host['RUNTIME_CACHE']), ROOT):
        parent = target
        while not parent.exists():
            parent = parent.parent
        extra = metadata['derived-cache']['minimum-free-gib'] if metadata['derived-cache'] else 30
        reserve = (host['DISK_RESERVE_GIB'] + host['PREPARE_OVERHEAD_GIB'] + extra) * 2**30
        if shutil.disk_usage(parent).free < reserve:
            fail(f'Insufficient disk reserve at {parent}')


def pull_image(image):
    print('Pulling candidate image...', flush=True)
    run(['docker', 'pull', '--platform', 'linux/arm64', image], capture=False)
    inspect = json.loads(run(['docker', 'image', 'inspect', image]))[0]
    if inspect['Architecture'] != 'arm64':
        fail('Candidate image is not native ARM64')
    digests = inspect.get('RepoDigests') or []
    if not digests:
        fail('Image has no immutable repository digest')
    return digests[0], inspect['Config'].get('Entrypoint')


def download_model_artifacts(host, secrets, pinned, repository, revision, kind='primary'):
    print(f'Downloading and hashing {repository}@{revision}...', flush=True)
    env = os.environ.copy()
    env['HF_TOKEN'] = secrets.get('HF_TOKEN', '')
    downloaded = run(['docker', 'run', '--rm', '--network', 'host', '--entrypoint', 'python3',
                      '--mount', f'type=bind,src={host["HF_CACHE"]},dst=/hf-cache',
                      '--mount', f'type=bind,src={ROOT / "lib/prepare_model.py"},dst=/prepare.py,readonly',
                      '--mount', f'type=bind,src={ROOT / "lib/artifact_layout.py"},dst=/artifact_layout.py,readonly',
                      '-e', 'HF_TOKEN', '-e', 'HTTP_PROXY', '-e', 'HTTPS_PROXY', '-e', 'NO_PROXY',
                      pinned, '/prepare.py', repository, revision, kind], env=env)
    marker = next((s.removeprefix('SPARK_MANIFEST=') for s in downloaded.splitlines()
                   if s.startswith('SPARK_MANIFEST=')), None)
    if marker is None:
        fail('Artifact verification did not return a manifest')
    return json.loads(marker)


def download_artifacts(host, secrets, profile, pinned):
    adapter = ADAPTERS[profile['engine']]
    repository, revision = adapter.repository(profile['native'])
    artifacts = download_model_artifacts(host, secrets, pinned, repository, revision)
    artifacts['auxiliary_models'] = []
    for model in profile['metadata'].get('auxiliary-models', []):
        manifest = download_model_artifacts(
            host, secrets, pinned, model['repository'], model['revision'], 'auxiliary')
        artifacts['auxiliary_models'].append({**model, **manifest})
    return artifacts


def local_manifest(host, manifest):
    try:
        relative = Path(manifest['snapshot']).relative_to('/hf-cache')
        snapshot = Path(host['HF_CACHE']) / relative
        entries = manifest['files']
    except (KeyError, TypeError, ValueError):
        fail('Downloaded-model manifest is invalid')
    if not isinstance(entries, list):
        fail('Downloaded-model manifest is invalid')
    return snapshot, entries


def manifest_available(host, manifest):
    snapshot, entries = local_manifest(host, manifest)
    return all((snapshot / entry['path']).is_file() and
               (snapshot / entry['path']).stat().st_size == entry['size'] for entry in entries)


def downloaded_artifacts(host, profile):
    metadata = profile['metadata']
    record = read_json(STATE / 'downloaded' / f'{metadata["id"]}.json')
    if not record:
        return None
    payload = record.get('payload')
    if not isinstance(payload, dict) or record.get('sha256') != digest_object(payload):
        fail('Downloaded-model record is malformed or was modified')
    repository, revision = ADAPTERS[profile['engine']].repository(profile['native'])
    if payload.get('profile') != metadata['id'] or payload.get('repository') != repository or \
            payload.get('revision') != revision:
        return None
    artifacts = payload.get('artifacts')
    if not manifest_available(host, artifacts):
        return None
    expected = profile['metadata'].get('auxiliary-models', [])
    auxiliary = artifacts.get('auxiliary_models', [])
    if not isinstance(auxiliary, list) or len(auxiliary) != len(expected):
        return None
    for wanted, manifest in zip(expected, auxiliary):
        if any(manifest.get(key) != wanted[key] for key in ('name', 'repository', 'revision')):
            return None
        if not manifest_available(host, manifest):
            return None
    return artifacts


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_auxiliary_artifacts(cache_path, metadata):
    artifacts = metadata.get('auxiliary-artifacts', [])
    if not artifacts:
        return
    target_dir = cache_path / 'auxiliary'
    target_dir.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        target = target_dir / artifact['name']
        if target.is_file() and file_sha256(target) == artifact['sha256']:
            continue
        fd, temporary_name = tempfile.mkstemp(dir=target_dir, prefix='.download-')
        try:
            with os.fdopen(fd, 'wb') as output:
                with urllib.request.urlopen(artifact['url'], timeout=120) as response:
                    shutil.copyfileobj(response, output)
                    output.flush()
                    os.fsync(output.fileno())
            temporary = Path(temporary_name)
            if file_sha256(temporary) != artifact['sha256']:
                fail(f'Auxiliary artifact hash mismatch: {artifact["name"]}')
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)


def verify_auxiliary_artifacts(rec):
    cache_path = Path(rec['compose_env']['RUNTIME_CACHE_DIR']) / 'auxiliary'
    for artifact in rec['identity'].get('profile_policy', {}).get('auxiliary-artifacts', []):
        target = cache_path / artifact['name']
        if not target.is_file() or file_sha256(target) != artifact['sha256']:
            fail(f'Prepared auxiliary artifact missing or invalid: {artifact["name"]}; rerun prepare')


def download(host, secrets, profile, image):
    metadata = profile['metadata']
    check_disk_capacity(host, metadata)
    pinned, _ = pull_image(image)
    artifacts = download_artifacts(host, secrets, profile, pinned)
    repository, revision = ADAPTERS[profile['engine']].repository(profile['native'])
    payload = {'profile': metadata['id'], 'repository': repository, 'revision': revision,
               'artifacts': artifacts, 'downloaded_at': time.time(), 'downloader_image': pinned}
    atomic(STATE / 'downloaded' / f'{metadata["id"]}.json',
           {'payload': payload, 'sha256': digest_object(payload)})
    total = sum(entry['size'] for entry in artifacts['files']) + sum(
        entry['size'] for model in artifacts.get('auxiliary_models', []) for entry in model['files'])
    print(f'Downloaded {metadata["id"]}: {total / 2**30:.1f} GiB verified at {artifacts["snapshot"]}.')
    return artifacts


def prepare(host, secrets, profile, image):
    metadata, model = profile['metadata'], profile['native']
    adapter = ADAPTERS[profile['engine']]
    adapter.validate(model, metadata)
    environment_report = doctor(host)
    check_disk_capacity(host, metadata)
    pinned, entrypoint = pull_image(image)
    adapter.inspect_entrypoint(entrypoint)
    cuda_code = adapter.cuda_probe()
    print('Validating a real CUDA operation in the pinned image...', flush=True)
    runtime = run(['docker', 'run', '--rm', '--gpus', 'all', '--entrypoint', 'python3', pinned, '-c', cuda_code])
    try:
        runtime = json.loads(runtime.splitlines()[-1])
    except (ValueError, IndexError):
        fail('CUDA probe did not produce valid runtime metadata')
    help_text = run(adapter.probe_command(pinned))
    if profile['engine'] == 'vllm' and metadata['runtime-environment'].get('VLLM_USE_RUST_FRONTEND'):
        adapter.validate_help(help_text, model, metadata,
                              rust_help=run(adapter.rust_probe_command(pinned)))
    else:
        adapter.validate_help(help_text, model, metadata)
    artifacts = downloaded_artifacts(host, profile)
    if artifacts:
        print(f'Using verified pre-downloaded snapshot for {metadata["id"]}.', flush=True)
    else:
        artifacts = download_artifacts(host, secrets, profile, pinned)
    auxiliary_models = {item['name']: item['snapshot']
                        for item in artifacts.get('auxiliary_models', [])}
    runtime_render = adapter.render(model, artifacts['snapshot'], host,
                                    authenticated=bool(secrets.get('VLLM_API_KEY')),
                                    auxiliary_models=auxiliary_models,
                                    rust_frontend=bool(
                                        metadata['runtime-environment'].get('VLLM_USE_RUST_FRONTEND')))
    config_text = runtime_render['config_text']
    compose_text = render_compose(runtime_render, metadata)
    runtime_files = runtime_render.get('files', {})
    if any(Path(name).name != name for name in runtime_files):
        fail('Invalid runtime file name')
    repository, revision = adapter.repository(model)
    policy = dict(metadata)
    identity = {'schema_version': 2, 'engine': profile['engine'], 'profile': metadata['id'],
                'image': pinned, 'repository': repository, 'revision': revision,
                'config': runtime_render['resolved'], 'profile_policy': policy,
                'policy_sha256': digest_object(policy), 'host': host,
                'invocation': {'entrypoint': runtime_render['entrypoint'], 'command': runtime_render['command']},
                'runtime_files_sha256': {name: hashlib.sha256(content.encode()).hexdigest()
                                         for name, content in runtime_files.items()},
                'compose_sha256': hashlib.sha256(compose_text.encode()).hexdigest()}
    release_id = digest_object(identity)[:24]
    path = STATE / 'releases' / release_id
    cache_path = Path(host['RUNTIME_CACHE']) / profile['engine'] / digest_object(
        {'image': pinned, 'config': runtime_render['resolved']})[:24]
    cache_path.mkdir(parents=True, exist_ok=True)
    ensure_auxiliary_artifacts(cache_path, metadata)
    compose_env = {'IMAGE': pinned, 'HF_CACHE': host['HF_CACHE'],
                   'RUNTIME_CACHE_DIR': str(cache_path), 'HEALTH_HOST': host['BIND_HOST'],
                   **{k: host[k] for k in ('PORT', 'HEALTH_TIMEOUT', 'STARTUP_TIMEOUT', 'STOP_GRACE_SECONDS')}}
    if metadata['derived-cache']:
        ple_path = Path(host['RUNTIME_CACHE']) / 'ple' / release_id
        ple_path.mkdir(parents=True, exist_ok=True)
        owner = ple_path / '.owner.json'
        if owner.exists() and read_json(owner) != {'release': release_id, 'kind': 'ple'}:
            fail('Derived-cache ownership marker does not match the release')
        if not owner.exists():
            if any(ple_path.iterdir()):
                fail('Refusing to claim a non-empty derived-cache directory')
            atomic(owner, {'release': release_id, 'kind': 'ple'})
        compose_env['PLE_CACHE_DIR'] = str(ple_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(dir=path.parent, prefix='.prepare-'))
        config_name = runtime_render['config_name']
        (temp / config_name).write_text(config_text)
        (temp / 'compose.yaml').write_text(compose_text)
        for name, content in runtime_files.items():
            (temp / name).write_text(content)
        rec = {'id': release_id, 'prepared_at': time.time(), 'identity': identity,
               'image_candidate': image, 'entrypoint': entrypoint, 'runtime': runtime,
               'artifacts': artifacts, 'environment': environment_report,
               'hashes': {n: hashlib.sha256((temp / n).read_bytes()).hexdigest()
                          for n in (config_name, 'compose.yaml', *runtime_files)},
               'compose_env': compose_env}
        atomic(temp / 'release.json', rec)
        os.rename(temp, path)
        for item in path.iterdir():
            item.chmod(0o444)
        path.chmod(0o555)
    release(release_id)
    compose(release_id, secrets, 'config', '--quiet')
    atomic(STATE / 'prepared' / f'{metadata["id"]}.json', {'profile': metadata['id'], 'release': release_id})
    if metadata['id'] == profiles.DEFAULT_PROFILE:
        atomic(STATE / 'prepared.json', {'release': release_id})
    print(f'Prepared {release_id} for {metadata["id"]}; CUDA passed. '
          f'Model execution and {metadata["context-tokens"]}-token qualification are pending.')
    return release_id


def state():
    return read_json(STATE / 'active.json', {'active': None, 'accepted': None, 'previous': None, 'running': False})


def test_release(release_id, secrets, mode):
    path, rec = release(release_id)
    env = os.environ.copy()
    env['VLLM_API_KEY'] = secrets.get('VLLM_API_KEY', '')
    return run([sys.executable, ROOT / 'tests/acceptance.py', '--release', str(path), '--mode', mode],
               env=env, capture=False)


def sustained_low_memory(low_since, available, minimum, now):
    if available >= minimum:
        return None, False
    if low_since is None:
        low_since = now
    return low_since, now - low_since >= MEMORY_GUARD_GRACE_SECONDS


def wait_ready(release_id, secrets):
    _, rec = release(release_id)
    host = rec['identity']['host']
    address = host['BIND_HOST']
    address = f'[{address}]' if ':' in address else address
    url = f'http://{address}:{host["PORT"]}/health'
    deadline = time.monotonic() + host['STARTUP_TIMEOUT']
    next_notice = 0
    minimum_memory = memory_available()
    low_memory_since = None
    memory_floor = host['MIN_AVAILABLE_GIB'] * 2**30
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        now = time.monotonic()
        available = memory_available()
        minimum_memory = min(minimum_memory, available)
        low_memory_since, failed = sustained_low_memory(low_memory_since, available, memory_floor, now)
        if failed:
            fail('Startup remained below the host available-memory guardrail for 15 seconds')
        rows = containers()
        if len(rows) != 1 or not rows[0]['State']['Running']:
            fail('Candidate exited during startup; inspect bin/spark-llm logs')
        try:
            request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + secrets.get('VLLM_API_KEY', '')})
            with opener.open(request, timeout=host['HEALTH_TIMEOUT']) as response:
                if response.status == 200:
                    atomic(ROOT / 'reports' / release_id / 'startup.json', {
                        'http_ready': True, 'minimum_mem_available_bytes': minimum_memory,
                        'time': time.time(), 'generation_test': 'separate'})
                    return
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() > next_notice:
            print('Waiting for model readiness...', flush=True)
            next_notice = time.monotonic() + 30
        time.sleep(3)
    fail('Startup timeout; configuration remains unqualified')



def swap_pages():
    values = {}
    for line in Path('/proc/vmstat').read_text().splitlines():
        key, value = line.split()
        if key in ('pswpin', 'pswpout'):
            values[key] = int(value)
    return values['pswpin'], values['pswpout']


def wait_for_swap_quiet(rec, quiet_seconds=10, timeout_seconds=600):
    deadline = time.monotonic() + timeout_seconds
    last = swap_pages()
    quiet_since = time.monotonic()
    low_memory_since = None
    memory_floor = rec['identity']['host']['MIN_AVAILABLE_GIB'] * 2**30
    while time.monotonic() < deadline:
        time.sleep(1)
        now = time.monotonic()
        current = swap_pages()
        if current != last:
            last = current
            quiet_since = now
        elif now - quiet_since >= quiet_seconds:
            return
        low_memory_since, failed = sustained_low_memory(
            low_memory_since, memory_available(), memory_floor, now)
        if failed:
            fail('Post-startup settling remained below the host available-memory guardrail')
    fail('Swap activity did not settle within 60 seconds after startup')


def stop(release_id, secrets):
    if release_id:
        compose(release_id, secrets, 'stop', capture=False)
    else:
        # Recover a project container left behind by an interrupted first launch.
        for row in containers():
            if row['State']['Running']:
                run(['docker', 'stop', '--time', '330', row['Id']])
    rows = containers()
    atomic(ROOT / 'reports/last-stop.json', {'release': release_id, 'time': time.time(),
           'forced_termination': any(c['State'].get('ExitCode') == 137 for c in rows)})


def launch(release_id, secrets):
    _, rec = release(release_id)
    address = ipaddress.ip_address(rec['identity']['host']['BIND_HOST'])
    if not address.is_loopback and not secrets.get('VLLM_API_KEY'):
        fail('A private-network bind requires VLLM_API_KEY in secrets.env')
    require_idle(rec['identity']['host'])
    reset_derived_cache(rec)
    # Offline Docker verification: never pull during start.
    run(['docker', 'image', 'inspect', rec['identity']['image']])
    verify_auxiliary_artifacts(rec)
    manifests = [rec['artifacts'], *rec['artifacts'].get('auxiliary_models', [])]
    for manifest in manifests:
        snapshot, entries = local_manifest(rec['identity']['host'], manifest)
        for entry in entries:
            item = snapshot / entry['path']
            if not item.is_file() or item.stat().st_size != entry['size']:
                fail(f'Prepared artifact missing/incomplete: {entry["path"]}; rerun prepare')
    compose(release_id, secrets, 'up', '-d', '--force-recreate', '--pull', 'never', capture=False)
    wait_ready(release_id, secrets)
    wait_for_swap_quiet(rec)
    test_release(release_id, secrets, 'smoke')
    backend_lines = []
    for row in containers():
        output = subprocess.run(['docker', 'logs', row['Id']], capture_output=True, text=True)
        for line in (output.stdout + output.stderr).splitlines():
            if re.search(r'backend|cache dtype|mamba|quantization|GPU KV cache|Maximum concurrency|Available KV cache|speculative|dflash|draft model', line, re.I):
                for value in secrets.values():
                    if value:
                        line = line.replace(value, '[REDACTED]')
                backend_lines.append(line[:4000])
    atomic(ROOT / 'reports' / release_id / 'runtime.json', {
        'release': release_id, 'runtime': rec['runtime'], 'engine_configuration_lines': backend_lines,
        'nvfp4_generation_smoke': 'passed', 'capacity_qualification': 'separate'})


def reset_derived_cache(rec):
    policy = rec['identity'].get('profile_policy', {}).get('derived-cache')
    if not policy or not policy.get('reset-before-start'):
        return
    if any(row['State']['Running'] for row in containers()):
        fail('Refusing to reset derived cache while the service is running')
    root = (Path(rec['identity']['host']['RUNTIME_CACHE']) / 'ple').resolve()
    path = Path(rec['compose_env']['PLE_CACHE_DIR'])
    if path.is_symlink() or path.resolve().parent != root:
        fail('Derived-cache path is outside the owned PLE root')
    owner = path / '.owner.json'
    if read_json(owner) != {'release': rec['id'], 'kind': 'ple'}:
        fail('Derived-cache ownership marker is missing or invalid')
    for item in path.iterdir():
        if item == owner:
            continue
        if item.is_symlink():
            fail('Refusing to follow a symlink in the derived cache')
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def replace(release_id, secrets):
    old = state()
    owned_running = any(row['State']['Running'] for row in containers())
    if old['active'] == release_id and old['running']:
        wait_ready(release_id, secrets)
        test_release(release_id, secrets, 'smoke')
        print('Requested release is already running.')
        return
    # Validate candidate fully before stopping the active service.
    _, candidate_record = release(release_id)
    address = ipaddress.ip_address(candidate_record['identity']['host']['BIND_HOST'])
    if not address.is_loopback and not secrets.get('VLLM_API_KEY'):
        fail('A private-network bind requires VLLM_API_KEY in secrets.env')
    compose(release_id, secrets, 'config', '--quiet')
    busy = conflicts()
    if busy:
        fail('Conflicting GPU containers: ' + ', '.join(busy))
    stop(old['active'], secrets)
    atomic(STATE / 'active.json', old | {'running': False})
    # A previous rollback can restore a healthy container but fail a
    # non-deterministic smoke assertion before updating active.json.
    fallback = old['accepted'] or (old['active'] if old['running'] or owned_running else None)
    if fallback == release_id:
        fallback = None
    try:
        launch(release_id, secrets)
    except Exception as error:
        candidate_logs = []
        for row in containers():
            output = subprocess.run(['docker', 'logs', '--tail', '500', row['Id']],
                                    capture_output=True, text=True)
            candidate_logs.extend((output.stdout + output.stderr).splitlines())
        for value in secrets.values():
            if value:
                candidate_logs = [line.replace(value, '[REDACTED]') for line in candidate_logs]
        atomic(ROOT / 'reports/last-failure.json', {'time': time.time(), 'candidate': release_id,
               'error': str(error), 'rollback_target': fallback,
               'candidate_logs': candidate_logs})
        stop(release_id, secrets)
        if fallback:
            launch(fallback, secrets)
            if fallback == old['accepted']:
                for c in containers():
                    run(['docker', 'update', '--restart', 'unless-stopped', c['Id']])
            atomic(STATE / 'active.json', old | {'active': fallback, 'running': True})
        raise
    atomic(STATE / 'active.json', old | {'active': release_id, 'running': True})
    qualified = read_json(STATE / 'qualification' / f'{release_id}.json', {})
    if qualified.get('daily_use') == 'accepted':
        for container in containers():
            run(['docker', 'update', '--restart', 'unless-stopped', container['Id']])
        atomic(STATE / 'active.json', old | {'active': release_id, 'running': True, 'accepted': release_id,
               'previous': old['accepted'] if old['accepted'] != release_id else old['previous']})
        print('Accepted release running; smoke passed and unless-stopped restored.')
    else:
        print('Running; smoke passed. Qualification pending; unattended restart is disabled until acceptance.')


def status():
    current = state()
    result = dict(current)
    if current['active']:
        _, rec = release(current['active'])
        identity = rec['identity']
        cfg = identity['config']
        engine = identity.get('engine', 'vllm')
        profile_id = identity.get('profile', profiles.DEFAULT_PROFILE)
        policy = identity.get('profile_policy', {})
        context_tokens = cfg['context-length'] if engine == 'sglang' else cfg['max-model-len']
        max_sequences = cfg['max-running-requests'] if engine == 'sglang' else cfg['max-num-seqs']
        result.update({'repository': rec['identity']['repository'], 'revision': rec['identity']['revision'],
                       'image_digest': rec['identity']['image'], 'configuration_hash': digest_object(cfg),
                       'alias': cfg['served-model-name'], 'max_model_len': context_tokens,
                       'max_num_seqs': max_sequences, 'engine': engine, 'profile': profile_id,
                       'configured_context_tokens': policy.get('context-tokens', context_tokens),
                       'qualification': read_json(STATE / 'qualification' / f'{current["active"]}.json', {'daily_use': 'pending'})})
        result['test_results'] = {mode: read_json(ROOT / 'reports' / current['active'] / f'{mode}.json', {}).get('passed', 'pending')
                                  for mode in ('smoke', 'api', 'context', 'benchmark', 'soak')}
    result['containers'] = [{'name': c['Name'], 'state': c['State']['Status'],
                             'health': c['State'].get('Health', {}).get('Status', 'unknown'),
                             'restart_count': c['RestartCount'], 'restart_policy': c['HostConfig']['RestartPolicy']['Name']}
                            for c in containers()]
    result['conflicting_containers'] = conflicts()
    prepared_dir = STATE / 'prepared'
    result['prepared'] = {p.stem: read_json(p) for p in sorted(prepared_dir.glob('*.json'))} \
        if prepared_dir.is_dir() else {}
    legacy = read_json(STATE / 'prepared.json')
    if legacy and profiles.DEFAULT_PROFILE not in result['prepared']:
        result['prepared'][profiles.DEFAULT_PROFILE] = legacy
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='status', choices=['doctor', 'profiles', 'download', 'prepare',
                        'start', 'stop', 'restart', 'status', 'logs', 'test', 'upgrade', 'rollback', 'validate', 'accept', 'help'])
    parser.add_argument('--profile', help=f'Model profile ID (default: {profiles.DEFAULT_PROFILE})')
    parser.add_argument('--release', help='Prepared release ID (default: last prepared)')
    parser.add_argument('--mode', choices=['smoke', 'api', 'context', 'benchmark', 'soak'], default='smoke')
    parser.add_argument('--disruptive', action='store_true', help='Allow long stress tests while clients are paused')
    parser.add_argument('--evidence', type=Path, help='Reviewed daily-use/recovery evidence JSON for acceptance')
    args = parser.parse_args()
    if args.command == 'help':
        parser.print_help()
        return
    cmd = args.command
    if cmd == 'validate':
        _, _, profile, _ = config(args.profile or profiles.DEFAULT_PROFILE)
        print(f'Profile {profile["metadata"]["id"]} is valid; hardware and runtime acceptance are separate.')
    elif cmd == 'profiles':
        active = state()['active']
        active_profile = None
        if active:
            active_profile = release(active)[1]['identity'].get('profile', profiles.DEFAULT_PROFILE)
        rows = [{'id': p['id'], 'engine': p['engine'], 'display_name': p['display-name'],
                 'context_tokens': p['context-tokens'], 'active': p['id'] == active_profile,
                 'prepared': (read_json(STATE / 'prepared' / f'{p["id"]}.json') or
                              (read_json(STATE / 'prepared.json')
                               if p['id'] == profiles.DEFAULT_PROFILE else None))}
                for p in profiles.list_profiles(ROOT)]
        print(json.dumps(rows, indent=2))
    elif cmd == 'doctor':
        host, _ = host_config()
        doctor(host)
    elif cmd == 'download':
        host, secrets, profile, image = config(args.profile or profiles.DEFAULT_PROFILE)
        download(host, secrets, profile, image)
    elif cmd == 'prepare':
        host, secrets, profile, image = config(args.profile or profiles.DEFAULT_PROFILE)
        prepare(host, secrets, profile, image)
    elif cmd == 'status':
        status()
    elif cmd == 'logs':
        secrets = secrets_config()
        # Avoid printing unrestricted Docker logs: redact credentials before display.
        rows = containers()
        for row in rows:
            p = subprocess.run(['docker', 'logs', '--tail', '150', row['Id']], capture_output=True, text=True)
            output = p.stdout + p.stderr
            for value in secrets.values():
                if value:
                    output = output.replace(value, '[REDACTED]')
            print(output)
    elif cmd in ('start', 'upgrade', 'rollback'):
        secrets = secrets_config()
        current = state()
        if cmd == 'rollback':
            candidate = current['previous']
        else:
            profile_id = args.profile or profiles.DEFAULT_PROFILE
            profiles.resolve(ROOT, profile_id)
            prepared = read_json(STATE / 'prepared' / f'{profile_id}.json')
            if prepared is None and profile_id == profiles.DEFAULT_PROFILE:
                prepared = read_json(STATE / 'prepared.json', {})
            candidate = args.release or (prepared or {}).get('release')
        if not candidate:
            fail('No prepared release for the selected profile or rollback target is available')
        if args.release and args.profile:
            _, selected = release(candidate)
            actual = selected['identity'].get('profile', profiles.DEFAULT_PROFILE)
            if actual != args.profile:
                fail(f'Release belongs to profile {actual}, not {args.profile}')
        replace(candidate, secrets)
    elif cmd == 'stop':
        secrets = secrets_config()
        current = state()
        stop(current['active'], secrets)
        atomic(STATE / 'active.json', current | {'running': False})
    elif cmd == 'restart':
        secrets = secrets_config()
        current = state()
        if not current['active']:
            fail('No active release')
        stop(current['active'], secrets)
        atomic(STATE / 'active.json', current | {'running': False})
        replace(current['active'], secrets)
    elif cmd == 'test':
        secrets = secrets_config()
        if args.mode in ('context', 'soak', 'benchmark') and not args.disruptive:
            fail('Use --disruptive with clients paused for long/performance tests')
        active = state()['active']
        if not active:
            fail('No active release')
        test_release(active, secrets, args.mode)
    elif cmd == 'accept':
        secrets = secrets_config()
        current = state()
        active = current['active']
        if not active or not args.evidence:
            fail('Acceptance requires an active release and --evidence reviewed.json')
        evidence = read_json(args.evidence)
        required = ['operator_accepted_responsiveness', 'recovery_suite_passed', 'regression_review_passed']
        if evidence.get('release') != active or not all(evidence.get(k) is True for k in required):
            fail('Reviewed evidence must identify this release and explicitly pass every acceptance field')
        for mode in ('api', 'context', 'benchmark', 'soak'):
            report = read_json(ROOT / 'reports' / active / f'{mode}.json', {})
            if report.get('passed') is not True:
                fail(f'{mode} has not passed for this release')
            if report.get('release') != active or report.get('mode') != mode:
                fail(f'{mode} report belongs to a different release or mode')
            identity = release(active)[1]['identity']
            if identity.get('schema_version', 1) >= 2:
                expected_harness = hashlib.sha256((ROOT / 'tests/acceptance.py').read_bytes()).hexdigest()
                if report.get('policy_sha256') != identity['policy_sha256'] or \
                        report.get('harness_sha256') != expected_harness:
                    fail(f'{mode} report policy or test harness identity does not match this release')
        test_release(active, secrets, 'smoke')
        accepted_context = release(active)[1]['identity'].get('profile_policy', {}).get(
            'context-tokens', release(active)[1]['identity']['config'].get('max-model-len'))
        atomic(STATE / 'qualification' / f'{active}.json', {'daily_use': 'accepted',
               'qualified_context_tokens': accepted_context, 'evidence': evidence, 'time': time.time()})
        for c in containers():
            run(['docker', 'update', '--restart', 'unless-stopped', c['Id']])
        atomic(STATE / 'active.json', current | {'accepted': active,
               'previous': current['accepted'] if current['accepted'] != active else current['previous']})
        print('Daily-use acceptance recorded; unless-stopped restart policy enabled.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
