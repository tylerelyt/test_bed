import { NextRequest } from 'next/server';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
async function proxy(request: NextRequest, context: { params: Promise<{path: string[]}> }) {
  const { path } = await context.params;
  // Only this server chooses the backend; credentials never enter browser storage.
  const base = process.env.WORKBENCH_API_URL || 'http://127.0.0.1:8766';
  if (request.method !== 'GET') {
    const origin = request.headers.get('origin');
    if (origin && new URL(origin).host !== request.headers.get('host')) return Response.json({error:'请求来源不匹配'}, {status:403});
  }
  try {
    if (path.join('/') === 'native/upload') {
      const size=Number(request.headers.get('X-File-Size'));
      if (!Number.isSafeInteger(size)||size<0) return Response.json({error:'文件长度无效'},{status:400});
      const init: RequestInit & {duplex:'half'} = {
        method:'POST', headers:{'Content-Type':'application/octet-stream',
          'Content-Length':String(size), 'X-File-Name':request.headers.get('X-File-Name')||'upload'},
        body:request.body, duplex:'half', signal:request.signal,
      };
      const upload=await fetch(`${base}/api/native/upload`,init);
      return new Response(upload.body,{status:upload.status,headers:{'Content-Type':'application/json','Cache-Control':'no-store'}});
    }
    const body = request.method === 'GET' ? undefined : await request.text();
    const limit = ['native/execute','native/event','native/runs'].includes(path.join('/')) ? 32_000_000 : 2_000_000;
    if (body && Buffer.byteLength(body) > limit) return Response.json({error:'请求体超过当前接口大小限制'}, {status:413});
    const response = await fetch(`${base}/api/${path.map(encodeURIComponent).join('/')}${request.nextUrl.search}`, {
      method: request.method, headers: {'Content-Type':'application/json'}, body,
      signal: request.signal, cache: 'no-store',
    });
    return new Response(response.body, {status:response.status, headers:{
      'Content-Type': response.headers.get('Content-Type') || 'application/json',
      'Cache-Control':'no-store', 'X-Accel-Buffering':'no',
    }});
  } catch {
    return Response.json({error:'Python 服务未连接。请先运行 python -m src.search_engine.workbench_api。'}, {status:503});
  }
}
export { proxy as GET, proxy as POST, proxy as DELETE };
