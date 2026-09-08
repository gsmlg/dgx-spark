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

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'state'
PROFILE = ROOT / 'profiles/qwen38-27b-nvfp4'
HOST_KEYS = {'HF_CACHE', 'RUNTIME_CACHE', 'BIND_HOST', 'PORT', 'STARTUP_TIMEOUT',
             'REQUEST_TIMEOUT', 'HEALTH_TIMEOUT', 'SHUTDOWN_TIMEOUT', 'STOP_GRACE_SECONDS',
             'DISK_RESERVE_GIB', 'PREPARE_OVERHEAD_GIB', 'MIN_AVAILABLE_GIB'}
MODEL_KEYS = {'model', 'revision', 'tokenizer-revision', 'served-model-name',
              'tensor-parallel-size', 'max-model-len', 'max-num-seqs', 'gpu-memory-utilization',
              'max-num-batched-tokens', 'enable-chunked-prefill', 'enable-prefix-caching',
              'dtype', 'kv-cache-dtype', 'language-model-only', 'reasoning-parser',
              'tool-call-parser', 'enable-auto-tool-choice', 'default-chat-template-kwargs',
              'generation-config', 'override-generation-config', 'enable-log-requests',
              'enable-log-outputs', 'disable-uvicorn-access-log'}


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


class UniqueLoader(yaml.SafeLoader):
    pass


def mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            fail(f'Duplicate YAML setting: {key}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)


def config():
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
    secrets = env_file(ROOT / 'secrets.env', {'HF_TOKEN', 'VLLM_API_KEY'}, False)
    if (ROOT / 'secrets.env').exists() and (ROOT / 'secrets.env').stat().st_mode & 0o077:
        fail('Run chmod 600 secrets.env before using credentials')
    if not address.is_loopback and not secrets.get('VLLM_API_KEY'):
        fail('A private-network bind requires VLLM_API_KEY in secrets.env')
    raw = (PROFILE / 'vllm.yaml').read_text()
    if '$' in raw:
        fail('Unresolved variable in model profile')
    model = yaml.load(raw, Loader=UniqueLoader)
    if not isinstance(model, dict) or set(model) != MODEL_KEYS:
        fail('Model profile has missing or unknown settings; host keys belong in host.env')
    if not re.fullmatch(r'[0-9a-f]{40}', str(model['revision'])):
        fail('Model revision must be a full 40-character commit')
    if model['tokenizer-revision'] != model['revision']:
        fail('Tokenizer and checkpoint revisions must match')
    for key in ('tensor-parallel-size', 'max-model-len', 'max-num-seqs', 'max-num-batched-tokens'):
        if type(model[key]) is not int or model[key] <= 0:
            fail(f'{key} must be a positive integer')
    for key in ('enable-chunked-prefill', 'enable-prefix-caching', 'language-model-only',
                'enable-auto-tool-choice', 'enable-log-requests', 'enable-log-outputs', 'disable-uvicorn-access-log'):
        if type(model[key]) is not bool:
            fail(f'{key} must be a YAML boolean')
    if type(model['gpu-memory-utilization']) not in (float, int):
        fail('gpu-memory-utilization must be numeric')
    if not isinstance(model['override-generation-config'], dict):
        fail('override-generation-config must be a mapping')
    if model['max-model-len'] != 262144 or model['max-num-seqs'] != 1:
        fail('This profile requires 262144 context and one scheduled sequence')
    if model['tensor-parallel-size'] != 1 or not 0 < model['gpu-memory-utilization'] < 1:
        fail('Invalid GPU allocation or tensor parallelism')
    if model['enable-log-requests'] or model['enable-log-outputs']:
        fail('Prompt and output logging must remain disabled')
    if not model['language-model-only']:
        fail('This profile must remain text-only')
    if 'max_new_tokens' in model['override-generation-config']:
        fail('Do not set a global completion cap through max_new_tokens')
    image = env_file(PROFILE / 'image.env', {'IMAGE'})
    if set(image) != {'IMAGE'} or not re.fullmatch(r'[a-zA-Z0-9._/@:-]+', image['IMAGE']):
        fail('image.env needs one literal IMAGE reference')
    return host, secrets, model, image['IMAGE']


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


def prepare(host, secrets, model, image):
    environment_report = doctor(host)
    for target in (Path(host['HF_CACHE']), Path(host['RUNTIME_CACHE']), ROOT):
        parent = target
        while not parent.exists():
            parent = parent.parent
        # Full snapshot plus conservative image/extraction/cache allowance.
        reserve = (host['DISK_RESERVE_GIB'] + host['PREPARE_OVERHEAD_GIB'] + 30) * 2**30
        if shutil.disk_usage(parent).free < reserve:
            fail(f'Insufficient disk reserve at {parent}')
    print('Pulling candidate image...', flush=True)
    run(['docker', 'pull', '--platform', 'linux/arm64', image], capture=False)
    inspect = json.loads(run(['docker', 'image', 'inspect', image]))[0]
    if inspect['Architecture'] != 'arm64':
        fail('Candidate image is not native ARM64')
    digests = inspect.get('RepoDigests') or []
    if not digests:
        fail('Image has no immutable repository digest')
    pinned = digests[0]
    entrypoint = inspect['Config'].get('Entrypoint')
    if entrypoint != ['vllm', 'serve']:
        fail(f'Unsupported image entrypoint {entrypoint}; inspect and adapt the common command explicitly')
    cuda_code = ('import torch,vllm,json; x=torch.ones((32,32),device="cuda"); '
                 'y=x@x; torch.cuda.synchronize(); assert y[0,0].item()==32; '
                 'print(json.dumps({"torch":torch.__version__,"vllm":vllm.__version__, '
                 '"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(), '
                 '"capability":torch.cuda.get_device_capability()}))')
    print('Validating a real CUDA operation in the pinned image...', flush=True)
    runtime = run(['docker', 'run', '--rm', '--gpus', 'all', '--entrypoint', 'python3', pinned, '-c', cuda_code])
    try:
        runtime = json.loads(runtime.splitlines()[-1])
    except (ValueError, IndexError):
        fail('CUDA probe did not produce valid runtime metadata')
    help_text = run(['docker', 'run', '--rm', '--gpus', 'all', pinned, '--help=all'])
    for key in model.keys() | {'host', 'port', 'shutdown-timeout'}:
        if '--' + key not in help_text:
            fail(f'Candidate runtime does not support --{key}')
    for parser in ('qwen3', 'qwen3_coder'):
        if parser not in help_text:
            fail(f'Candidate runtime does not list parser {parser}')
    print('Downloading and hashing the complete pinned snapshot...', flush=True)
    env = os.environ.copy()
    env['HF_TOKEN'] = secrets.get('HF_TOKEN', '')
    downloaded = run(['docker', 'run', '--rm', '--network', 'host', '--entrypoint', 'python3',
                      '--mount', f'type=bind,src={host["HF_CACHE"]},dst=/hf-cache',
                      '--mount', f'type=bind,src={ROOT / "lib/prepare_model.py"},dst=/prepare.py,readonly',
                      '-e', 'HF_TOKEN', '-e', 'HTTP_PROXY', '-e', 'HTTPS_PROXY', '-e', 'NO_PROXY',
                      pinned, '/prepare.py', model['model'], model['revision']], env=env)
    marker = next((s.removeprefix('SPARK_MANIFEST=') for s in downloaded.splitlines()
                   if s.startswith('SPARK_MANIFEST=')), None)
    if marker is None:
        fail('Artifact verification did not return a manifest')
    artifacts = json.loads(marker)
    resolved = dict(model)
    resolved.update({'model': artifacts['snapshot'], 'tokenizer': artifacts['snapshot'],
                     'host': host['BIND_HOST'], 'port': host['PORT'],
                     'shutdown-timeout': host['SHUTDOWN_TIMEOUT']})
    config_text = yaml.safe_dump(resolved, sort_keys=False)
    identity = {'image': pinned, 'repository': model['model'], 'revision': model['revision'],
                'config': resolved, 'host': host, 'compose_sha256': hashlib.sha256((ROOT / 'compose.yaml').read_bytes()).hexdigest()}
    release_id = digest_object(identity)[:24]
    path = STATE / 'releases' / release_id
    cache_path = Path(host['RUNTIME_CACHE']) / digest_object({'image': pinned, 'config': resolved})[:24]
    cache_path.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(dir=path.parent, prefix='.prepare-'))
        (temp / 'vllm.yaml').write_text(config_text)
        shutil.copyfile(ROOT / 'compose.yaml', temp / 'compose.yaml')
        rec = {'id': release_id, 'prepared_at': time.time(), 'identity': identity,
               'image_candidate': image, 'entrypoint': entrypoint, 'runtime': runtime,
               'artifacts': artifacts, 'environment': environment_report,
               'hashes': {n: hashlib.sha256((temp / n).read_bytes()).hexdigest() for n in ('vllm.yaml', 'compose.yaml')},
               'compose_env': {'IMAGE': pinned, 'HF_CACHE': host['HF_CACHE'],
                               'RUNTIME_CACHE_DIR': str(cache_path), 'HEALTH_HOST': host['BIND_HOST'],
                               **{k: host[k] for k in ('PORT', 'HEALTH_TIMEOUT', 'STARTUP_TIMEOUT', 'STOP_GRACE_SECONDS')}}}
        atomic(temp / 'release.json', rec)
        os.rename(temp, path)
        for item in path.iterdir():
            item.chmod(0o444)
        path.chmod(0o555)
    release(release_id)
    compose(release_id, secrets, 'config', '--quiet')
    atomic(STATE / 'prepared.json', {'release': release_id})
    print(f'Prepared {release_id}; CUDA passed. Model execution and 256K qualification are pending.')
    return release_id


def state():
    return read_json(STATE / 'active.json', {'active': None, 'accepted': None, 'previous': None, 'running': False})


def test_release(release_id, secrets, mode):
    path, rec = release(release_id)
    env = os.environ.copy()
    env['VLLM_API_KEY'] = secrets.get('VLLM_API_KEY', '')
    return run([sys.executable, ROOT / 'tests/acceptance.py', '--release', str(path), '--mode', mode],
               env=env, capture=False)


def wait_ready(release_id, secrets):
    _, rec = release(release_id)
    host = rec['identity']['host']
    address = host['BIND_HOST']
    address = f'[{address}]' if ':' in address else address
    url = f'http://{address}:{host["PORT"]}/health'
    deadline = time.monotonic() + host['STARTUP_TIMEOUT']
    next_notice = 0
    minimum_memory = memory_available()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        minimum_memory = min(minimum_memory, memory_available())
        if minimum_memory < host['MIN_AVAILABLE_GIB'] * 2**30:
            fail('Startup crossed the host available-memory guardrail')
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
    require_idle(rec['identity']['host'])
    # Offline Docker verification: never pull during start.
    run(['docker', 'image', 'inspect', rec['identity']['image']])
    snapshot = Path(rec['identity']['host']['HF_CACHE']) / Path(rec['artifacts']['snapshot']).relative_to('/hf-cache')
    for entry in rec['artifacts']['files']:
        item = snapshot / entry['path']
        if not item.is_file() or item.stat().st_size != entry['size']:
            fail(f'Prepared artifact missing/incomplete: {entry["path"]}; rerun prepare')
    compose(release_id, secrets, 'up', '-d', '--force-recreate', '--pull', 'never', capture=False)
    wait_ready(release_id, secrets)
    test_release(release_id, secrets, 'smoke')
    backend_lines = []
    for row in containers():
        output = subprocess.run(['docker', 'logs', row['Id']], capture_output=True, text=True)
        for line in (output.stdout + output.stderr).splitlines():
            if re.search(r'backend|cache dtype|mamba|quantization|GPU KV cache|Maximum concurrency|Available KV cache', line, re.I):
                for value in secrets.values():
                    if value:
                        line = line.replace(value, '[REDACTED]')
                backend_lines.append(line[:4000])
    atomic(ROOT / 'reports' / release_id / 'runtime.json', {
        'release': release_id, 'runtime': rec['runtime'], 'engine_configuration_lines': backend_lines,
        'nvfp4_generation_smoke': 'passed', 'capacity_qualification': 'separate'})


def replace(release_id, secrets):
    old = state()
    if old['active'] == release_id and old['running']:
        wait_ready(release_id, secrets)
        test_release(release_id, secrets, 'smoke')
        print('Requested release is already running.')
        return
    # Validate candidate fully before stopping the active service.
    release(release_id)
    compose(release_id, secrets, 'config', '--quiet')
    busy = conflicts()
    if busy:
        fail('Conflicting GPU containers: ' + ', '.join(busy))
    stop(old['active'], secrets)
    atomic(STATE / 'active.json', old | {'running': False})
    try:
        launch(release_id, secrets)
    except Exception as error:
        atomic(ROOT / 'reports/last-failure.json', {'time': time.time(), 'candidate': release_id,
               'error': str(error), 'rollback_target': old['accepted']})
        stop(release_id, secrets)
        if old['accepted'] and old['accepted'] != release_id:
            launch(old['accepted'], secrets)
            for c in containers():
                run(['docker', 'update', '--restart', 'unless-stopped', c['Id']])
            atomic(STATE / 'active.json', old | {'active': old['accepted'], 'running': True})
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
        cfg = rec['identity']['config']
        result.update({'repository': rec['identity']['repository'], 'revision': rec['identity']['revision'],
                       'image_digest': rec['identity']['image'], 'configuration_hash': digest_object(cfg),
                       'alias': cfg['served-model-name'], 'max_model_len': cfg['max-model-len'],
                       'max_num_seqs': cfg['max-num-seqs'],
                       'qualification': read_json(STATE / 'qualification' / f'{current["active"]}.json', {'daily_use': 'pending'})})
        result['test_results'] = {mode: read_json(ROOT / 'reports' / current['active'] / f'{mode}.json', {}).get('passed', 'pending')
                                  for mode in ('smoke', 'api', 'context', 'benchmark', 'soak')}
    result['containers'] = [{'name': c['Name'], 'state': c['State']['Status'],
                             'health': c['State'].get('Health', {}).get('Status', 'unknown'),
                             'restart_count': c['RestartCount'], 'restart_policy': c['HostConfig']['RestartPolicy']['Name']}
                            for c in containers()]
    result['conflicting_containers'] = conflicts()
    result['prepared'] = read_json(STATE / 'prepared.json')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='status', choices=['doctor', 'prepare', 'start', 'stop',
                        'restart', 'status', 'logs', 'test', 'upgrade', 'rollback', 'validate', 'accept', 'help'])
    parser.add_argument('--release', help='Prepared release ID (default: last prepared)')
    parser.add_argument('--mode', choices=['smoke', 'api', 'context', 'benchmark', 'soak'], default='smoke')
    parser.add_argument('--disruptive', action='store_true', help='Allow long stress tests while clients are paused')
    parser.add_argument('--evidence', type=Path, help='Reviewed daily-use/recovery evidence JSON for acceptance')
    args = parser.parse_args()
    if args.command == 'help':
        parser.print_help()
        return
    host, secrets, model, image = config()
    cmd = args.command
    if cmd == 'validate':
        print('Configuration valid; hardware and runtime acceptance are separate.')
    elif cmd == 'doctor':
        doctor(host)
    elif cmd == 'prepare':
        prepare(host, secrets, model, image)
    elif cmd == 'status':
        status()
    elif cmd == 'logs':
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
        current = state()
        if cmd == 'rollback':
            candidate = current['previous']
        else:
            candidate = args.release or read_json(STATE / 'prepared.json', {}).get('release')
        if not candidate:
            fail('No prepared release or rollback target is available')
        replace(candidate, secrets)
    elif cmd == 'stop':
        current = state()
        stop(current['active'], secrets)
        atomic(STATE / 'active.json', current | {'running': False})
    elif cmd == 'restart':
        current = state()
        if not current['active']:
            fail('No active release')
        stop(current['active'], secrets)
        atomic(STATE / 'active.json', current | {'running': False})
        replace(current['active'], secrets)
    elif cmd == 'test':
        if args.mode in ('context', 'soak', 'benchmark') and not args.disruptive:
            fail('Use --disruptive with clients paused for long/performance tests')
        active = state()['active']
        if not active:
            fail('No active release')
        test_release(active, secrets, args.mode)
    elif cmd == 'accept':
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
        test_release(active, secrets, 'smoke')
        atomic(STATE / 'qualification' / f'{active}.json', {'daily_use': 'accepted', 'evidence': evidence, 'time': time.time()})
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
