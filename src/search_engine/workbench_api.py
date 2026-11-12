"""Loopback HTTP transport for the original workbench's native Next.js UI."""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote, quote
from .native_runtime import runtime


def make_handler(executor=runtime):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def send_json(self, data, status=200):
            body = json.dumps(executor.export(data), ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/api/health':
                return self.send_json({'status': 'ready', 'modules_initialized': executor.blocks is not None,
                    'engine': 'original-python-callbacks', 'separate_gradio_server': False})
            if path.startswith('/api/native/assets/'):
                asset = executor.assets.get(path.rsplit('/', 1)[-1])
                if not asset or not asset.is_file():
                    return self.send_json({'error': '文件已过期，请重新运行'}, 404)
                self.send_response(200)
                self.send_header('Content-Type', mimetypes.guess_type(asset.name)[0] or 'application/octet-stream')
                self.send_header('Content-Length', str(asset.stat().st_size))
                self.send_header('Content-Disposition', "inline; filename*=UTF-8''" + quote(asset.name))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    with asset.open('rb') as source:
                        while chunk := source.read(1024 * 1024):
                            self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_json({'error': '接口不存在'}, 404)

        def do_POST(self):
            if self.headers.get('Origin'):
                return self.send_json({'error': '请通过控制台同源接口访问'}, 403)
            path = urlparse(self.path).path
            if path == '/api/native/upload':
                try:
                    size = int(self.headers.get('Content-Length', '-1'))
                    if size < 0:
                        raise ValueError('文件长度无效')
                    name = unquote(self.headers.get('X-File-Name', 'upload'))
                    result = executor.store_upload(name, self.rfile, size)
                    return self.send_json(result)
                except (ValueError, OSError) as exc:
                    self.close_connection = True
                    return self.send_json({'error': str(exc)}, 400)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                limit = 32_000_000 if path == '/api/native/event' else 2_000_000
                if not 0 < size <= limit:
                    return self.send_json({'error': '请求体大小无效'}, 413)
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError('请求必须是 JSON 对象')
                session_id = data.get('session')
                if not isinstance(session_id, str) or not session_id or len(session_id) > 120:
                    raise ValueError('会话标识无效')
                if path == '/api/native/layout':
                    try:
                        return self.send_json(executor.layout(session_id))
                    except Exception as exc:
                        return self.send_json({'error': '原模块初始化失败：' + str(exc)}, 503)
                if path != '/api/native/event':
                    return self.send_json({'error': '接口不存在'}, 404)
                fn_id = data.get('fn_id')
                if isinstance(fn_id, bool) or not isinstance(fn_id, int) or not isinstance(data.get('values', {}), dict):
                    raise ValueError('事件或输入格式无效')
                # Resolve the actual registry before starting the response stream.
                executor.initialize()
                if fn_id not in executor.blocks.fns:
                    raise ValueError('操作不存在')
            except (ValueError, TypeError) as exc:
                return self.send_json({'error': str(exc)}, 400)
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
            self.send_header('Connection', 'close')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.close_connection = True
            def emit(item):
                try:
                    self.wfile.write((json.dumps(executor.export(item), ensure_ascii=False, allow_nan=False) + '\n').encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            try:
                executor.execute(session_id, fn_id, data.get('values', {}), data.get('event'), emit=emit)
            except Exception as exc:
                emit({'type': 'failed', 'error': str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description='原 Python 模块的 Next.js 执行服务')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler())
    print(f'Python API: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == '__main__':
    main()
