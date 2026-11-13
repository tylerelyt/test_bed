"""Local protocol fixture, not an intelligent model or a production provider."""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ModelHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(body)
        self.server.received.set()
        if self.server.pause:
            self.server.release.wait(timeout=15)
        tool_result = next((m for m in reversed(body['messages']) if m.get('role') == 'tool'), None)
        if tool_result is None:
            delta = {'role': 'assistant', 'tool_calls': [{'index': 0, 'id': 'fixture_context_call',
                     'type': 'function', 'function': {'name': 'read_enterprise_context', 'arguments': '{}'}}]}
            finish = 'tool_calls'
        else:
            # Echo actual tool output so an empty or incorrectly scoped context fails acceptance.
            delta = {'role': 'assistant', 'content': '已读取预配置上下文，依据如下：\n' + tool_result['content']}
            finish = 'stop'
        message = {'id': 'fixture-completion', 'object': 'chat.completion.chunk', 'created': int(time.time()),
                   'model': body['model'], 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]}
        done = {**message, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}],
                'usage': {'prompt_tokens': 40, 'completion_tokens': 20, 'total_tokens': 60}}
        data = ('data: ' + json.dumps(message, ensure_ascii=False) + '\n\n' +
                'data: ' + json.dumps(done) + '\n\ndata: [DONE]\n\n').encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


def server(port=0):
    import threading
    instance = ThreadingHTTPServer(('127.0.0.1', port), ModelHandler)
    instance.requests = []
    instance.received, instance.release = threading.Event(), threading.Event()
    instance.pause = False
    instance.daemon_threads = True
    return instance


if __name__ == '__main__':
    server(8779).serve_forever()
