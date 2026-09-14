#!/usr/bin/env python3
"""Synthetic API/capacity tests. Never execute model-suggested tools."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
from lifecycle import atomic, containers, memory_available

SWAP_GUARD_GRACE_SECONDS = 15


def resource_guardrail_failed(resources, minimum_available_bytes):
    return (resources['minimum_mem_available_bytes'] < minimum_available_bytes or
            resources['oom_kills'] > 0 or
            resources['longest_continuous_swap_seconds'] >= SWAP_GUARD_GRACE_SECONDS)


class Client:
    def __init__(self, release):
        self.release = json.loads((release / 'release.json').read_text())
        host = self.release['identity']['host']
        address = host['BIND_HOST']
        address = f'[{address}]' if ':' in address else address
        self.base = f'http://{address}:{host["PORT"]}'
        self.timeout = host['REQUEST_TIMEOUT']
        self.alias = self.release['identity']['config']['served-model-name']
        identity = self.release['identity']
        config = identity['config']
        self.policy = identity.get('profile_policy')
        if self.policy is None:
            self.policy = {
                'id': 'qwen38-27b-nvfp4', 'engine': 'vllm',
                'context-tokens': config['max-model-len'], 'reasoning-default': False,
                'reasoning-toggle': True, 'reasoning-fields': ['reasoning', 'reasoning_content'],
                'tool-support': True}
        self.context_tokens = self.policy['context-tokens']
        self.records = []
        self.key = os.environ.get('VLLM_API_KEY', '')
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def open(self, path, body=None):
        req = urllib.request.Request(self.base + path,
            data=json.dumps(body, ensure_ascii=False).encode() if body is not None else None,
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key})
        return self.opener.open(req, timeout=self.timeout)

    def json(self, path, body=None):
        with self.open(path, body) as response:
            return json.load(response)

    def body(self, prompt='Reply with a short greeting.', **kwargs):
        return {'model': self.alias, 'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 128, 'n': 1, **kwargs}

    def count(self, messages):
        body = {'model': self.alias, 'messages': messages, 'add_generation_prompt': True}
        if self.policy['reasoning-toggle']:
            body['chat_template_kwargs'] = {'enable_thinking': self.policy['reasoning-default']}
        result = self.json('/tokenize', body)
        return result['count']

    def generate(self, label, body=None, stream=False):
        body = body or self.body()
        started = time.monotonic()
        first = None
        content = ''
        reasoning = ''
        streamed_calls = {}
        if stream:
            body = body | {'stream': True, 'stream_options': {'include_usage': True}}
            done = False
            usage = None
            with self.open('/v1/chat/completions', body) as response:
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    data = line[6:].strip()
                    if data == b'[DONE]':
                        done = True
                        break
                    event = json.loads(data)
                    if event.get('usage'):
                        usage = event['usage']
                    for choice in event.get('choices', []):
                        delta = choice.get('delta', {})
                        text = delta.get('content') or ''
                        thought = delta.get('reasoning') or delta.get('reasoning_content') or ''
                        calls = delta.get('tool_calls') or []
                        if (text or thought or calls) and first is None:
                            first = time.monotonic()
                        content += text
                        reasoning += thought
                        for call in calls:
                            index = call.get('index', 0)
                            current = streamed_calls.setdefault(index, {
                                'id': '', 'type': 'function',
                                'function': {'name': '', 'arguments': ''}})
                            current['id'] += call.get('id') or ''
                            current['type'] = call.get('type') or current['type']
                            function = call.get('function') or {}
                            current['function']['name'] += function.get('name') or ''
                            current['function']['arguments'] += function.get('arguments') or ''
            assert done and usage is not None, 'Stream missing DONE or usage'
            message = {'role': 'assistant', 'content': content, 'reasoning': reasoning}
            if streamed_calls:
                message['tool_calls'] = [streamed_calls[index] for index in sorted(streamed_calls)]
            result = {'usage': usage, 'choices': [{'message': message}]}
        else:
            result = self.json('/v1/chat/completions', body)
            content = result['choices'][0]['message'].get('content') or ''
        elapsed = time.monotonic() - started
        usage = result['usage']
        record = {'case': label, 'requested_output_tokens': body['max_tokens'],
                  'prompt_tokens': usage['prompt_tokens'], 'generated_tokens': usage['completion_tokens'],
                  'latency_seconds': elapsed, 'ttft_seconds': first - started if first else None,
                  'decode_tokens_per_second': ((usage['completion_tokens'] - 1) / (time.monotonic() - first)
                                               if first and usage['completion_tokens'] > 1 else None),
                  'message_fields': list(result['choices'][0]['message'])}
        self.records.append(record)
        print(json.dumps(record), flush=True)
        return result

    def expect_error(self, body, label):
        try:
            self.json('/v1/chat/completions', body)
        except urllib.error.HTTPError as error:
            assert 400 <= error.code < 500, f'{label}: expected client error, got {error.code}'
            self.records.append({'case': label, 'http_status': error.code})
            return
        raise AssertionError(f'{label}: request was accepted unexpectedly')

    def tool(self, name='lookup_temperature', argument='city', value='Hong Kong', answer='23', stream=False):
        tool = {'type': 'function', 'function': {'name': name,
                'description': 'Look up the requested information using the provided key.',
                'parameters': {'type': 'object', 'properties': {argument: {'type': 'string'}},
                               'required': [argument], 'additionalProperties': False}}}
        body = self.body(f'Use {name} to look up {argument} {value}. Then report its result.',
                         tools=[tool], tool_choice='auto', max_tokens=512)
        result = self.generate('tool-' + name, body, stream=stream)
        message = result['choices'][0]['message']
        calls = message.get('tool_calls')
        assert calls and len(calls) == 1, 'Expected one tool call'
        call = calls[0]
        assert call['id'] and call['function']['name'] == name
        arguments = json.loads(call['function']['arguments'])
        assert arguments.get(argument) == value, 'Incorrect tool arguments'
        history_fields = {'role', 'content', 'tool_calls', *self.policy['reasoning-fields']}
        body['messages'] += [{k: v for k, v in message.items() if k in history_fields},
                             {'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps({'result': answer})}]
        body['tool_choice'] = 'none'
        final = self.generate('tool-result-' + name, body)
        assert answer in response_text(final['choices'][0]['message']), 'Tool result not used'


def response_text(message):
    return ''.join(str(message.get(key) or '') for key in ('content', 'reasoning', 'reasoning_content'))


def smoke(c):
    models = c.json('/v1/models')['data']
    assert any(m['id'] == c.alias for m in models), 'Expected alias missing'
    result = c.generate('english')
    message = result['choices'][0]['message']
    assert response_text(message), 'Empty response'
    reasoning = any(message.get(field) for field in c.policy['reasoning-fields'])
    assert reasoning == c.policy['reasoning-default'], 'Default reasoning behavior does not match the profile'
    assert '<think>' not in (message.get('content') or ''), 'Parser artifact in answer'
    streamed = c.generate('stream', stream=True)
    assert streamed['choices'][0]['message']['content']
    c.tool(stream=True)


def cancellation(c, messages=None, max_tokens=None):
    max_tokens = max_tokens or min(16384, max(512, c.context_tokens // 4))
    body = c.body('Write a very long explanation of sorting algorithms.', max_tokens=max_tokens,
                  stream=True, ignore_eos=True)
    if messages:
        body['messages'] = messages
    with c.open('/v1/chat/completions', body) as response:
        for line in response:
            if line.startswith(b'data: ') and b'content' in line:
                break
    recovered = c.generate('post-cancellation')
    assert recovered['choices'][0]['message'].get('content')


def api(c):
    smoke(c)
    result = c.generate('chinese', c.body('请用中文简短介绍你可以如何帮助编程。'))
    text = result['choices'][0]['message'].get('content') or ''
    assert any('\u4e00' <= ch <= '\u9fff' for ch in text)
    c.tool('lookup_inventory', 'sku', 'ITEM-17', '42')
    c.tool('lookup_status', 'ticket', 'TASK-83', 'complete')
    if c.policy['reasoning-toggle']:
        thought = c.generate('thinking', c.body('What is 37 multiplied by 49? Explain briefly.',
                             max_tokens=1024, chat_template_kwargs={'enable_thinking': True},
                             temperature=1.0, top_p=0.95, top_k=20, presence_penalty=0))
        msg = thought['choices'][0]['message']
        assert any(msg.get(field) for field in c.policy['reasoning-fields']), 'Reasoning field is empty'
        assert '1813' in response_text(msg)
    c.expect_error(c.body(model='missing-model'), 'unknown-alias')
    c.expect_error({'model': c.alias, 'messages': 'invalid'}, 'malformed')
    c.expect_error(c.body(messages=[{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Describe the image.'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo='}}]}]), 'unsupported-media')
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda i: c.generate(f'overlap-{i}'), range(2)))
    assert all(r['choices'][0]['message'].get('content') for r in results)
    cancellation(c)


def fixture(c, target, seed):
    """Exact formatted count via the deployed tokenizer/template, with varied records."""
    rng = random.Random(seed)
    values = [f'{rng.getrandbits(48):012x}' for _ in range(3)]
    rows = []
    # Enough varied text for >262K tokens; no repeated single-token capacity fixture.
    for i in range(40000):
        rows.append(f'Record {i}: station {rng.randrange(100000)} observed {rng.choice(["cedar", "quartz", "river", "copper"])} '
                    f'with measurement {rng.randrange(1000000)} and reference {rng.getrandbits(40):010x}.\n')
    filler = ''.join(rows)
    def messages(n):
        text = filler[:n]
        mid = len(text) // 2
        content = (f'BEGIN_KEY={values[0]}\n' + text[:mid] + f'\nMIDDLE_KEY={values[1]}\n' + text[mid:]
                   + f'\nEND_KEY={values[2]}\nReturn the exact BEGIN_KEY, MIDDLE_KEY and END_KEY values, each labeled.')
        return [{'role': 'system', 'content': f'Fixture {seed}. Retrieve facts exactly from the supplied record.'},
                {'role': 'user', 'content': content}]
    lo, hi = 0, len(filler)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if c.count(messages(mid)) <= target - 32:
            lo = mid
        else:
            hi = mid - 1
    result = messages(lo)
    # Small suffix adjustment; every final candidate is counted, never estimated.
    for _ in range(100):
        count = c.count(result)
        if count == target:
            return result, values
        if count > target:
            raise AssertionError('Fixture adjustment overshot target')
        result[-1]['content'] += ' x' * max(1, target - count)
    raise AssertionError('Could not construct exact formatted token count')


def context(c):
    combined = c.context_tokens
    large_output = min(16384, max(1024, combined // 8))
    targets = sorted(set((min(4096, combined - 512), combined // 2, combined - large_output)))
    for target in targets:
        messages, _ = fixture(c, target, target)
        budget = large_output if target == combined - large_output else 512
        result = c.generate(f'input-{target}', c.body(messages=messages, max_tokens=budget,
                            cache_salt=uuid.uuid4().hex), stream=True)
        assert result['usage']['prompt_tokens'] == target, 'Server/template count mismatch'
        assert result['choices'][0]['message'].get('content')
        c.generate('post-long-recovery')
    boundary_input = combined - 1024
    for seed in (117, 229, 331):
        messages, values = fixture(c, boundary_input, seed)
        result = c.generate(f'near-limit-seed-{seed}-cold-prefix', c.body(messages=messages,
                            max_tokens=1024, cache_salt=uuid.uuid4().hex), stream=True)
        assert result['usage']['prompt_tokens'] == boundary_input
        answer = response_text(result['choices'][0]['message'])
        assert all(value in answer for value in values), 'Retrieval quality failed'
        c.generate('post-near-limit')
    # Clearly separated forced-length engine stress, never daily generation defaults.
    result = c.generate('forced-boundary-stress-not-quality', c.body(messages=messages,
                        max_tokens=1024, min_tokens=1024, ignore_eos=True,
                        cache_salt=uuid.uuid4().hex), stream=True)
    assert result['usage']['prompt_tokens'] == boundary_input and result['usage']['completion_tokens'] == 1024
    c.generate('post-stress')
    c.expect_error(c.body(messages=messages, max_tokens=1025), 'invalid-combined-budget')
    oversized, _ = fixture(c, combined + 1, 443)
    c.expect_error(c.body(messages=oversized, max_tokens=1), 'oversized-prompt')
    cancellation(c, messages, max_tokens=1024)


def benchmark(c):
    messages, _ = fixture(c, 4096, 555)
    c.generate('short-warmup', c.body(messages=messages, max_tokens=512), stream=True)
    for n in range(20):
        c.generate(f'warm-short-{n}', c.body(messages=messages, max_tokens=512), stream=True)


def soak(c):
    deadline = time.monotonic() + 8 * 3600
    long_input = min(32768, c.context_tokens - 512)
    messages, _ = fixture(c, long_input, 888)
    for n in range(100):
        assert time.monotonic() < deadline, 'Workload exceeded eight-hour window'
        c.generate(f'soak-short-{n}', stream=True)
        if n % 10 == 0:
            c.tool()
        if n in (10, 50, 90):
            c.generate(f'soak-long-{n}', c.body(messages=messages, max_tokens=512), stream=True)
        remaining = deadline - time.monotonic()
        # Spread requests throughout the window; caller prints periodic progress.
        pause = max(0, remaining / (100 - n))
        while pause > 0:
            interval = min(30, pause)
            time.sleep(interval)
            pause -= interval
            print(f'Soak observation: {max(0, int(deadline-time.monotonic()))} seconds remaining', flush=True)


class Monitor:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop.is_set():
            vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
            self.samples.append({'time': time.time(), 'available': memory_available(),
                                 **{k: int(vm[k]) for k in ('pswpin', 'pswpout', 'oom_kill')}})
            self.stop.wait(1)

    def finish(self):
        self.stop.set()
        self.thread.join()
        if not self.samples:
            raise AssertionError('No memory samples recorded')
        first, last = self.samples[0], self.samples[-1]
        longest = current = 0
        for a, b in zip(self.samples, self.samples[1:]):
            current = current + 1 if b['pswpin'] > a['pswpin'] or b['pswpout'] > a['pswpout'] else 0
            longest = max(longest, current)
        return {'minimum_mem_available_bytes': min(x['available'] for x in self.samples),
                'oom_kills': last['oom_kill'] - first['oom_kill'],
                'swap_in_pages': last['pswpin'] - first['pswpin'],
                'swap_out_pages': last['pswpout'] - first['pswpout'],
                'longest_continuous_swap_seconds': longest, 'sample_count': len(self.samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--mode', choices=['smoke', 'api', 'context', 'benchmark', 'soak'], default='smoke')
    args = parser.parse_args()
    c = Client(args.release)
    monitor = Monitor()
    monitor.thread.start()
    before = [(x['Id'], x['RestartCount']) for x in containers()]
    harness_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {'release': c.release['id'], 'mode': args.mode,
              'policy_sha256': c.release['identity'].get('policy_sha256'),
              'harness_sha256': harness_sha256, 'started_at': time.time(), 'passed': False}
    try:
        globals()[args.mode](c)
        after = [(x['Id'], x['RestartCount']) for x in containers()]
        assert before == after and len(after) == 1, 'Container restarted or was replaced during test'
        report['passed'] = True
    except Exception as error:
        # Never serialize exception bodies, request prompts or responses.
        report['failure_type'] = type(error).__name__
        if isinstance(error, AssertionError):
            report['failure'] = str(error)
    finally:
        report['resources'] = monitor.finish()
        res = report['resources']
        minimum = c.release['identity']['host']['MIN_AVAILABLE_GIB'] * 2**30
        if resource_guardrail_failed(res, minimum):
            report['passed'] = False
            report['memory_guardrail_failed'] = True
        report['finished_at'] = time.time()
        report['cases'] = c.records
        if args.mode == 'benchmark' and report['passed']:
            rows = [r for r in c.records if r['case'].startswith('warm-short-')]
            report['summary'] = {key: {'median': statistics.median(r[key] for r in rows),
                'minimum': min(r[key] for r in rows), 'maximum': max(r[key] for r in rows)}
                for key in ('ttft_seconds', 'decode_tokens_per_second', 'latency_seconds')}
        target = ROOT / 'reports' / c.release['id'] / f'{args.mode}.json'
        atomic(target, report)
        # Retain every run, including failures, as well as the latest result.
        atomic(target.with_name(f'{args.mode}-{time.time_ns()}.json'), report)
        print(f'{args.mode}: {"PASS" if report["passed"] else "FAIL"}; report: {target}', flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
