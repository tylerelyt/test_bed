import { NextRequest } from 'next/server';
import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
async function proxy(request: NextRequest, context: { params: Promise<{path: string[]}> }) {
  const { path } = await context.params;
  // Like research-desk, the console server owns the credential. The browser
  // cannot select a different backend identity through a header or cookie.
  const base = process.env.WORKBENCH_API_URL || 'http://127.0.0.1:8766';
  if (request.method !== 'GET') {
    const origin = request.headers.get('origin');
    if (origin) {
      try {
        if (new URL(origin).host !== request.headers.get('host')) return Response.json({error:'请求来源不匹配'}, {status:403});
      } catch {
        return Response.json({error:'请求来源无效'}, {status:403});
      }
    }
    if (path.join('/') !== 'native/upload' && !request.headers.get('content-type')?.includes('application/json'))
      return Response.json({error:'此操作需要 JSON 请求'}, {status:415});
  }
  let credential: string;
  try {
    const state = process.env.WORKBENCH_STATE_DIR || resolve(process.cwd(), '../.workbench');
    credential = process.env.WORKBENCH_CONSOLE_TOKEN?.trim() ||
      (await readFile(/* turbopackIgnore: true */ process.env.WORKBENCH_TOKEN_FILE || join(state, 'bootstrap-access.txt'), 'utf8')).trim();
    if (!credential || /[\r\n]/.test(credential)) throw Error('invalid server credential');
  } catch {
    return Response.json({error:'服务端访问配置未就绪，请检查工作台启动配置。'}, {status:503});
  }
  const identityHeaders = {Authorization:'Bearer ' + credential};
  try {
    if (path.join('/') === 'native/upload') {
      const sizeHeader=request.headers.get('X-File-Size'),size=Number(sizeHeader);
      if (sizeHeader===null||!Number.isSafeInteger(size)||size<0) return Response.json({error:'文件长度无效'},{status:400});
      const init: RequestInit & {duplex:'half'} = {
        method:'POST', headers:{...identityHeaders,'Content-Type':'application/octet-stream',
          'Content-Length':String(size), 'X-File-Name':request.headers.get('X-File-Name')||'upload'},
        body:request.body, duplex:'half', signal:request.signal,
      };
      const upload=await fetch(`${base}/api/native/upload`,init);
      return new Response(upload.body,{status:upload.status,headers:{'Content-Type':'application/json','Cache-Control':'no-store'}});
    }
    const body = request.method === 'GET' ? undefined : await request.text();
    const limit = ['native/execute','native/event','native/runs'].includes(path.join('/')) ? 32_000_000 : ['platform/datasets','platform/jobs'].includes(path.join('/')) ? 14_000_000 : 2_000_000;
    if (body && Buffer.byteLength(body) > limit) return Response.json({error:'请求体超过当前接口大小限制'}, {status:413});
    const response = await fetch(`${base}/api/${path.map(encodeURIComponent).join('/')}${request.nextUrl.search}`, {
      method: request.method, headers: {...identityHeaders,'Content-Type':'application/json'}, body,
      signal: request.signal, cache: 'no-store',
    });
    return new Response(response.body, {status:response.status, headers:{
      'Content-Type': response.headers.get('Content-Type') || 'application/json',
      'Cache-Control':'no-store', 'X-Accel-Buffering':'no',
      ...(response.headers.get('Content-Disposition')?{'Content-Disposition':response.headers.get('Content-Disposition')!}:{}),
      ...(response.headers.get('Content-Security-Policy')?{'Content-Security-Policy':response.headers.get('Content-Security-Policy')!}:{}),
      'X-Content-Type-Options':'nosniff',
    }});
  } catch {
    return Response.json({error:'平台后端未连接。请运行 ./start_console.sh，或检查 platform_api 与 platform_worker 日志。'}, {status:503});
  }
}
export { proxy as GET, proxy as POST, proxy as DELETE };
