"""Narrow Responses compatibility for Mistral's chat renderer and tool names."""

import hashlib
import json
import re


VALID_TOOL_NAME = re.compile(r'[A-Za-z0-9_-]{1,64}\Z')


def _short_alias(name, maximum, reserved, prefix):
    salt = 0
    while True:
        digest = hashlib.sha256(f'{name}\0{salt}'.encode()).hexdigest()
        alias = prefix + digest[:maximum - len(prefix)]
        if alias not in reserved:
            reserved.add(alias)
            return alias
        salt += 1


def _tool_aliases(request):
    """Build request-local aliases, including vLLM's namespace__function names."""
    names = set()
    namespaces = {}
    tools = request.get('tools')
    for tool in tools if isinstance(tools, list) else []:
        if not isinstance(tool, dict):
            continue
        name = tool.get('name')
        if not isinstance(name, str):
            continue
        if tool.get('type') == 'function':
            names.add(name)
        elif tool.get('type') == 'namespace':
            children = namespaces.setdefault(name, set())
            nested_tools = tool.get('tools')
            for child in nested_tools if isinstance(nested_tools, list) else []:
                if (isinstance(child, dict) and child.get('type') == 'function'
                        and isinstance(child.get('name'), str)):
                    children.add(child['name'])

    functions = {}
    namespace_aliases = {}
    children_aliases = {}
    reserved_names = names | {child for children in namespaces.values() for child in children}
    reserved_namespaces = set(namespaces)
    reserved_flat = names | {f'{namespace}__{child}' for namespace, children in namespaces.items()
                             for child in children}
    for name in sorted(names):
        if not VALID_TOOL_NAME.fullmatch(name):
            functions[name] = _short_alias(name, 64, reserved_names, 'fn_')
    for namespace, children in sorted(namespaces.items()):
        if not children:
            continue
        short_namespace = namespace
        if (not VALID_TOOL_NAME.fullmatch(namespace)
                or any(not VALID_TOOL_NAME.fullmatch(f'{namespace}__{child}')
                       for child in children) and 64 - len(namespace) - 2 < 8):
            short_namespace = _short_alias(namespace, 19, reserved_namespaces, 'ns_')
            namespace_aliases[namespace] = short_namespace
        for child in sorted(children):
            flat = f'{short_namespace}__{child}'
            if VALID_TOOL_NAME.fullmatch(flat) and flat not in reserved_flat - {
                    f'{namespace}__{child}'}:
                continue
            maximum = 64 - len(short_namespace) - 2
            short_child = _short_alias(f'{namespace}__{child}', maximum,
                                       reserved_names, 'fn_')
            children_aliases[(namespace, child)] = short_child
            reserved_flat.add(f'{short_namespace}__{short_child}')

    if not (functions or namespace_aliases or children_aliases):
        return {}
    pairs = {}
    for namespace, children in namespaces.items():
        for child in children:
            short_namespace = namespace_aliases.get(namespace, namespace)
            short_child = children_aliases.get((namespace, child), child)
            if (short_namespace, short_child) != (namespace, child):
                pairs[(namespace, child)] = (short_namespace, short_child)
    flats = {f'{namespace}__{child}': f'{short_namespace}__{short_child}'
             for (namespace, child), (short_namespace, short_child) in pairs.items()}
    return {'functions': functions, 'namespaces': namespace_aliases,
            'children': children_aliases, 'pairs': pairs, 'flats': flats}


def _rewrite_function_names(value, aliases, parent_namespace=None):
    """Rewrite only typed function references, not arbitrary name fields or text."""
    if isinstance(value, list):
        return [_rewrite_function_names(item, aliases, parent_namespace) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get('type')
    namespace = value.get('name') if kind == 'namespace' else parent_namespace
    result = {key: _rewrite_function_names(item, aliases, namespace) for key, item in value.items()}
    if kind == 'namespace' and isinstance(value.get('name'), str):
        result['name'] = aliases['namespaces'].get(value['name'], value['name'])
    elif kind in ('function', 'function_call') and isinstance(value.get('name'), str):
        if isinstance(value.get('namespace'), str):
            namespace = value['namespace']
            result['namespace'] = aliases['namespaces'].get(namespace, namespace)
        if namespace is not None:
            result['name'] = aliases['children'].get((namespace, value['name']), value['name'])
        else:
            result['name'] = aliases['functions'].get(
                value['name'], aliases['flats'].get(value['name'], value['name']))
    return result


def _reverse_aliases(aliases):
    if not aliases:
        return {}
    names = {short: original for original, short in aliases['functions'].items()}
    children = {short: original for original, short in aliases['pairs'].items()}
    for (_, original_child), short_child in aliases['children'].items():
        names[short_child] = original_child
    return {'names': names,
            'namespaces': {short: original for original, short in aliases['namespaces'].items()},
            'children': children,
            'flats': {short: original for original, short in aliases['flats'].items()}}


def _rewrite_request(body):
    try:
        request = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body, {}
    if not isinstance(request, dict):
        return body, {}
    aliases = _tool_aliases(request)
    changed = bool(aliases)
    if aliases:
        request = {**request}
        for field in ('tools', 'tool_choice', 'input'):
            if field in request:
                request[field] = _rewrite_function_names(request[field], aliases)
    if not isinstance(request.get('input'), list):
        if not changed:
            return body, {}
        return json.dumps(request, ensure_ascii=False, separators=(',', ':')).encode(), _reverse_aliases(aliases)
    messages = []
    for message in request['input']:
        if not isinstance(message, dict) or 'role' not in message:
            messages.append(message)
            continue
        normalized = message
        if message['role'] == 'developer':
            normalized = {**normalized, 'role': 'system'}
            changed = True
        content = message.get('content')
        if (isinstance(content, list) and content
                and all(isinstance(part, dict) and set(part) == {'type', 'text'}
                        and part['type'] == 'input_text' and isinstance(part['text'], str)
                        for part in content)):
            normalized = {**normalized, 'content': ''.join(part['text'] for part in content)}
            changed = True
        messages.append(normalized)
    if not changed:
        return body, {}
    return json.dumps({**request, 'input': messages}, ensure_ascii=False,
                      separators=(',', ':')).encode(), _reverse_aliases(aliases)


def rewrite_responses_messages(body):
    """Map developer roles, text content, and invalid function names."""
    return _rewrite_request(body)[0]


def _restore_function_names(value, aliases):
    if isinstance(value, list):
        return [_restore_function_names(item, aliases) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _restore_function_names(item, aliases) for key, item in value.items()}
    kind = value.get('type')
    name = value.get('name')
    if kind == 'function_call' and isinstance(name, str):
        namespace = value.get('namespace')
        if isinstance(namespace, str):
            original = aliases['children'].get((namespace, name))
            if original:
                result['namespace'], result['name'] = original
            else:
                result['namespace'] = aliases['namespaces'].get(namespace, namespace)
        else:
            result['name'] = aliases['names'].get(name, aliases['flats'].get(name, name))
    elif kind == 'response.function_call_arguments.done' and isinstance(name, str):
        result['name'] = aliases['names'].get(name, aliases['flats'].get(name, name))
    return result


def _restore_json(body, aliases):
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body
    restored = _restore_function_names(value, aliases)
    if restored == value:
        return body
    return json.dumps(restored, ensure_ascii=False, separators=(',', ':')).encode()


def _restore_sse_frame(frame, aliases):
    match = re.search(rb'\r?\n\r?\n\Z', frame)
    payload, separator = (frame[:match.start()], frame[match.start():]) if match else (frame, b'')
    lines = payload.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(b'data:'):
            data = line[5:].strip()
            restored = _restore_json(data, aliases)
            if restored != data:
                ending = b'\r\n' if line.endswith(b'\r\n') else b'\n' if line.endswith(b'\n') else b''
                lines[index] = b'data: ' + restored + ending
    return b''.join(lines) + separator


class ResponsesMessageMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (scope['type'] != 'http' or scope.get('method') != 'POST'
                or scope.get('path') != '/v1/responses'):
            return await self.app(scope, receive, send)

        events = []
        chunks = []
        while True:
            event = await receive()
            events.append(event)
            if event['type'] != 'http.request':
                break
            chunks.append(event.get('body', b''))
            if not event.get('more_body', False):
                break

        if events[-1]['type'] == 'http.request':
            body = b''.join(chunks)
            rewritten, aliases = _rewrite_request(body)
            if rewritten != body:
                events = [{'type': 'http.request', 'body': rewritten, 'more_body': False}]
                scope = dict(scope)
                scope['headers'] = [
                    (name, str(len(rewritten)).encode() if name.lower() == b'content-length' else value)
                    for name, value in scope.get('headers', [])]
        else:
            aliases = {}

        async def replay_receive():
            if events:
                return events.pop(0)
            return await receive()

        if not aliases:
            return await self.app(scope, replay_receive, send)

        response_start = None
        response_chunks = []
        pending_sse = b''
        streaming = False

        async def restore_send(event):
            nonlocal response_start, pending_sse, streaming
            if event['type'] == 'http.response.start':
                response_start = event
                headers = event.get('headers', [])
                streaming = any(name.lower() == b'content-type'
                                and b'text/event-stream' in value.lower()
                                for name, value in headers)
                if streaming:
                    await send({**event, 'headers': [
                        (name, value) for name, value in headers
                        if name.lower() != b'content-length']})
                return
            if event['type'] != 'http.response.body' or response_start is None:
                return await send(event)
            if streaming:
                pending_sse += event.get('body', b'')
                frames = []
                while True:
                    match = re.search(rb'\r?\n\r?\n', pending_sse)
                    if not match:
                        break
                    end = match.end()
                    frames.append(_restore_sse_frame(pending_sse[:end], aliases))
                    pending_sse = pending_sse[end:]
                if not event.get('more_body', False):
                    frames.append(_restore_sse_frame(pending_sse, aliases))
                    pending_sse = b''
                await send({**event, 'body': b''.join(frames)})
                return
            response_chunks.append(event.get('body', b''))
            if event.get('more_body', False):
                return
            restored = _restore_json(b''.join(response_chunks), aliases)
            headers = [
                (name, str(len(restored)).encode() if name.lower() == b'content-length' else value)
                for name, value in response_start.get('headers', [])]
            await send({**response_start, 'headers': headers})
            await send({**event, 'body': restored})

        return await self.app(scope, replay_receive, restore_send)
