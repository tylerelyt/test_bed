"""Single management API with the legacy callback transport retained as an adapter."""
import argparse
import mimetypes
import threading
import os
from pathlib import Path
from types import SimpleNamespace
from contextlib import asynccontextmanager
from http.server import ThreadingHTTPServer

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from sqlalchemy.exc import IntegrityError

from .agent_platform.service import AgentPlatform
from .workbench_access import AccessDenied, require_business, require_operator


def create_app(executor=None, records=None, database_url=None, embedded_worker=False, external_worker=False, legacy_index=None):
    @asynccontextmanager
    async def lifespan(app):
        if external_worker:
            from .workbench_access import AccessPolicy
            from .workbench_context import ContextStore
            directory=Path(os.environ.get('WORKBENCH_STATE_DIR') or Path(__file__).resolve().parents[2]/'.workbench')
            active_records=SimpleNamespace(directory=directory,access=AccessPolicy(directory),context=ContextStore(directory))
            platform=AgentPlatform(directory,active_records.context,active_records.access,database_url,background=False,legacy_index=legacy_index)
            app.state.records,app.state.platform=active_records,platform
            async with httpx.AsyncClient(timeout=None) as client:
                app.state.client=client
                app.state.legacy='http://127.0.0.1:'+os.environ.get('WORKBENCH_BUSINESS_PORT','8767')
                yield
            platform.close()
            return
        from .native_runtime import runtime
        from .workbench_api import make_handler
        from .workbench_runs import RunStore
        active_executor = executor or runtime
        active_records = records or RunStore(active_executor, background=True)
        platform = AgentPlatform(active_records.directory, active_records.context, active_records.access, database_url,
                                 background=embedded_worker,legacy_index=legacy_index)
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
                if request.method != 'GET':
                    parts, size = [], 0
                    async for part in request.stream():
                        size += len(part)
                        if size > 14_000_000:
                            return JSONResponse({'error': '请求内容超过 14 MB'}, status_code=413)
                        parts.append(part)
                    request._body = b''.join(parts)
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

    @app.exception_handler(ImportError)
    async def missing_dependency(request,exc):
        return JSONResponse({'error':'业务服务依赖未就绪，请安装 requirements-context.txt；图谱等原功能另需完整依赖'},status_code=503)

    @app.exception_handler(AccessDenied)
    async def denied(request, exc):
        return JSONResponse({'error': str(exc)}, status_code=exc.status)

    @app.exception_handler(IntegrityError)
    async def conflict(request, exc):
        return JSONResponse({'error': '记录版本或请求标识冲突，请刷新后核对'}, status_code=409)

    @app.get('/api/platform/overview')
    def overview():
        return app.state.platform.overview()

    @app.get('/api/platform/environment')
    def environment():
        return app.state.platform.environment()

    @app.get('/api/platform/context')
    def contexts(q: str = ''):
        return app.state.platform.context.catalog(q)

    @app.post('/api/platform/context-supply/validate')
    def validate_supply(data:dict):
        return app.state.platform.validation.rag(data.get('index_id'),data.get('model_id'),data.get('query'))

    @app.post('/api/platform/context-supply/import')
    def import_context(data:dict):
        return app.state.platform.knowledge.import_file(data,app.state.platform.jobs)

    @app.get('/api/platform/context-supply/indexes')
    def context_indexes():
        return app.state.platform.knowledge.listing()

    @app.get('/api/platform/context-supply/sources/legacy-index')
    def legacy_documents():
        return app.state.platform.knowledge.legacy_documents()

    @app.post('/api/platform/context-supply/sources/legacy-graph')
    def graph_facts(data:dict):
        return app.state.platform.knowledge.snapshot_graph(data.get('entity'),data.get('name'))

    @app.post('/api/platform/context-supply/sources/legacy-index')
    def source_document(data:dict):
        return app.state.platform.knowledge.snapshot_legacy(data.get('document_id'),data.get('name'))

    @app.post('/api/platform/context-supply/indexes')
    def build_index(data:dict):
        return app.state.platform.knowledge.build(data.get('name'),data.get('resources'))

    @app.post('/api/platform/context-supply/indexes/{index_id}/search')
    def search_index(index_id:str,data:dict):
        return app.state.platform.knowledge.search(index_id,data.get('query'),data.get('limit',5))

    @app.post('/api/platform/context-supply/indexes/{index_id}/reference')
    def reference_index(index_id:str,data:dict):
        return app.state.platform.knowledge.reference(index_id,data.get('name'))

    @app.get('/api/platform/context/{table}/{item_id}')
    def context_item(table: str, item_id: str, version: int | None = None):
        if table not in ('resources', 'bundles'):
            raise ValueError('上下文类型无效')
        return app.state.platform.context.get(table, item_id, version)

    @app.post('/api/platform/context')
    def context_action(data: dict):
        store = app.state.platform.context
        action = data.get('action')
        if action == 'resource':
            return store.resource(data.get('name'), data.get('kind'), data.get('content'), data.get('source', ''),
                data.get('permission', 'read_only'), data.get('tags'), data.get('resource_id'), data.get('expected_version'))
        if action == 'bundle':
            return store.bundle(data.get('name'), data.get('resources'), data.get('max_chars', 12000),
                data.get('bundle_id'), data.get('expected_version'))
        if action == 'assemble':
            return store.assemble(data.get('bundle_id'), data.get('version'), data.get('resource_ids'), data.get('query', ''))
        raise ValueError('上下文操作无效')

    @app.post('/api/platform/datasets')
    def dataset(data: dict):
        return app.state.platform.models.dataset(data.get('name'), data.get('rows'),data.get('stage','sft'))

    @app.post('/api/platform/model-supply/train')
    def train_model(data:dict):
        return app.state.platform.models.train(data,app.state.platform.jobs)

    @app.post('/api/platform/model-supply/{operation}')
    def model_operation(operation:str,data:dict):
        if operation not in ('merge','deploy'): raise ValueError('模型操作不存在')
        return app.state.platform.models.start_operation(operation,data,app.state.platform.jobs)

    @app.get('/api/platform/model-supply')
    def model_supply():
        return app.state.platform.models.listing()

    @app.post('/api/platform/artifacts')
    def artifact(data: dict):
        return app.state.platform.models.artifact(data)

    @app.post('/api/platform/jobs')
    def job(data: dict):
        return app.state.platform.jobs.start(data)

    @app.get('/api/platform/jobs')
    def jobs():
        return sorted(app.state.platform.jobs.listing()+app.state.platform.business.listing(),key=lambda x:x.get('created_at',0),reverse=True)

    def business_log(run_id):
        path=app.state.platform.directory/'business-processes'/run_id/'process.log'
        if not path.is_file():return '业务事件与输出保存在统一数据库；详细回放见模块运行记录。'
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-32000))
            return stream.read(32000).decode('utf-8',errors='replace')

    @app.get('/api/platform/jobs/{job_id}')
    async def job_detail(job_id: str, request: Request):
        if job_id.startswith('business:'):
            response=await app.state.client.get(app.state.legacy+'/api/native/runs/'+job_id[9:],
                headers={'Authorization':request.headers.get('authorization','')})
            if response.status_code!=200:return JSONResponse(response.json(),status_code=response.status_code)
            run=response.json()
            return {'id':job_id,'name':run['action'],'kind':'business','status':run['status'],
                'error':run.get('error',''),'result':{'outputs':run.get('outputs',{}),'artifacts':run.get('artifacts',[]),
                'context':run.get('context'),'model_service':run.get('model_service'),'training':run.get('training')},
                'log':business_log(run['id'])}
        return app.state.platform.jobs.detail(job_id)

    @app.post('/api/platform/jobs/{job_id}/cancel')
    async def job_cancel(job_id: str, data: dict, request: Request):
        if job_id.startswith('business:'):
            response=await app.state.client.post(app.state.legacy+'/api/native/runs/control',
                headers={'Authorization':request.headers.get('authorization','')},json={'run_id':job_id[9:],'action':'cancel'})
            if response.status_code!=200:return JSONResponse(response.json(),status_code=response.status_code)
            return await job_detail(job_id,request)
        return app.state.platform.jobs.cancel(job_id)

    @app.post('/api/platform/business-runs/{run_id}/publish')
    def publish_business(run_id: str, data: dict):
        return JSONResponse({'error':'已撤除界面输出发布。请从上下文供应选择真实资料或索引来源。'},status_code=410)

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
        try:
            response = await app.state.client.send(upstream, stream=True)
        except httpx.RequestError:
            return JSONResponse({'error':'业务执行进程未连接，请检查 platform_worker 日志'},status_code=503)
        return StreamingResponse(response.aiter_raw(), status_code=response.status_code,
            headers={key:value for key,value in response.headers.items() if key.lower() not in ('connection', 'transfer-encoding', 'server', 'date')},
            background=BackgroundTask(response.aclose))
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description='岗位 Agent 平台与原模块兼容服务')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    uvicorn.run(create_app(external_worker=True), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
