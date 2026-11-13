"""Single management API with the legacy callback transport retained as an adapter."""
import argparse
import mimetypes
import threading
from contextlib import asynccontextmanager
from http.server import ThreadingHTTPServer

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from sqlalchemy.exc import IntegrityError

from .agent_platform.service import AgentPlatform
from .workbench_access import AccessDenied, require_business, require_operator


def create_app(executor=None, records=None, database_url=None, embedded_worker=False):
    @asynccontextmanager
    async def lifespan(app):
        from .native_runtime import runtime
        from .workbench_api import make_handler
        from .workbench_runs import RunStore
        active_executor = executor or runtime
        active_records = records or RunStore(active_executor, background=True)
        platform = AgentPlatform(active_records.directory, active_records.context, active_records.access, database_url,
                                 background=embedded_worker)
        legacy = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(active_executor, active_records))
        legacy.daemon_threads = True
        thread = threading.Thread(target=legacy.serve_forever, daemon=True)
        thread.start()
        app.state.records, app.state.platform = active_records, platform
        async with httpx.AsyncClient(timeout=None) as client:
            app.state.client = client
            app.state.legacy = f'http://127.0.0.1:{legacy.server_port}'
            yield
        platform.close()
        legacy.shutdown()
        legacy.server_close()
        thread.join(timeout=3)
        if records is None:
            active_records.close()

    app = FastAPI(title='岗位 Agent 与企业上下文供应后台', lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.middleware('http')
    async def authenticate(request, call_next):
        if request.url.path.startswith('/api/platform/'):
            try:
                if request.method != 'GET' and request.headers.get('origin'):
                    raise AccessDenied('请通过控制台同源接口访问')
                policy = app.state.records.access
                with policy.bind(policy.authenticate(request.headers.get('authorization'))):
                    require_operator()
                    return await call_next(request)
            except AccessDenied as exc:
                return JSONResponse({'error': str(exc)}, status_code=exc.status)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({'error': str(exc)}, status_code=400)

    @app.exception_handler(AccessDenied)
    async def denied(request, exc):
        return JSONResponse({'error': str(exc)}, status_code=exc.status)

    @app.exception_handler(IntegrityError)
    async def conflict(request, exc):
        return JSONResponse({'error': '记录版本或请求标识冲突，请刷新后核对'}, status_code=409)

    @app.get('/api/platform/overview')
    def overview():
        return app.state.platform.overview()

    @app.post('/api/platform/positions')
    def position(data: dict):
        return app.state.platform.catalog.position(data.get('name'), data.get('duties'))

    @app.post('/api/platform/agents')
    def agent(data: dict):
        return app.state.platform.catalog.agent(data.get('position_id'), data.get('name'))

    @app.post('/api/platform/models')
    def model(data: dict):
        return app.state.platform.catalog.model(data)

    @app.post('/api/platform/releases')
    def release(data: dict):
        return app.state.platform.catalog.release(data.get('agent_id'), data.get('model_id'), data.get('bundle_id'), data.get('bundle_version'))

    @app.post('/api/platform/releases/{release_id}/publish')
    def publish(release_id: str, data: dict):
        return app.state.platform.catalog.publish(release_id, data.get('run_id'))

    @app.post('/api/platform/runs')
    def start(data: dict):
        return app.state.platform.runs.start(data.get('agent_id'), data.get('release_id'), data.get('task'), data.get('request_id'))

    @app.get('/api/platform/runs/{run_id}')
    def run(run_id: str, after: int = 0):
        if after < 0:
            raise ValueError('事件游标无效')
        return app.state.platform.runs.detail(run_id, after)

    @app.post('/api/platform/runs/{run_id}/cancel')
    def cancel(run_id: str, data: dict):
        return app.state.platform.runs.cancel(run_id)

    @app.get('/api/platform/runs/{run_id}/result')
    def result(run_id: str):
        detail = app.state.platform.runs.detail(run_id)
        path = app.state.platform.directory / 'agent-workspaces' / detail['id'] / 'result.txt'
        if detail['status'] != 'completed' or not path.is_file():
            raise ValueError('本次运行暂无已完成的结果文件')
        return FileResponse(path, media_type='text/plain; charset=utf-8', filename='岗位任务结果.txt')

    @app.api_route('/{path:path}', methods=['GET', 'POST', 'DELETE'])
    async def legacy(request: Request, path: str):
        headers = {key: value for key, value in request.headers.items() if key.lower() not in ('host', 'connection', 'transfer-encoding')}
        upstream = app.state.client.build_request(request.method, app.state.legacy + request.url.path +
                   ('?' + request.url.query if request.url.query else ''), headers=headers,
                   content=request.stream() if request.method != 'GET' else None)
        response = await app.state.client.send(upstream, stream=True)
        return StreamingResponse(response.aiter_raw(), status_code=response.status_code,
            headers={key:value for key,value in response.headers.items() if key.lower() not in ('connection', 'transfer-encoding', 'server', 'date')},
            background=BackgroundTask(response.aclose))
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description='岗位 Agent 平台与原模块兼容服务')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    uvicorn.run(create_app(), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
