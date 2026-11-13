import { NextRequest, NextResponse } from 'next/server';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
const cookie = 'workbench_access';
async function inspect(token: string) {
  const base = process.env.WORKBENCH_API_URL || 'http://127.0.0.1:8766';
  const response = await fetch(base + '/api/native/access', {
    headers: token ? {Authorization: 'Bearer ' + token} : {}, cache: 'no-store',
  });
  const value = await response.json();
  if (!response.ok) throw Error(value.error || '身份服务不可用');
  return value;
}
export async function GET(request: NextRequest) {
  try {
    return NextResponse.json(await inspect(request.cookies.get(cookie)?.value || ''), {headers:{'Cache-Control':'no-store'}});
  } catch (error) {
    return NextResponse.json({error:(error as Error).message}, {status:503});
  }
}
function sameOrigin(request: NextRequest) {
  const origin = request.headers.get('origin');
  return !origin || new URL(origin).host === request.headers.get('host');
}
export async function POST(request: NextRequest) {
  if (!sameOrigin(request)) return NextResponse.json({error:'请求来源不匹配'}, {status:403});
  try {
    const reader = request.body?.getReader();
    const chunks:Uint8Array[]=[];let size=0;
    if(reader)while(true){
      const item=await reader.read();if(item.done)break;
      size+=item.value.byteLength;
      if(size>4096){await reader.cancel();return NextResponse.json({error:'凭证请求过大'},{status:413})}
      chunks.push(item.value);
    }
    const body=Buffer.concat(chunks).toString('utf8');
    const {token} = JSON.parse(body);
    if (typeof token !== 'string' || token.length < 24 || token.length > 2048)
      return NextResponse.json({error:'请输入有效的访问凭证'}, {status:400});
    const result = await inspect(token);
    if (!result.authenticated) return NextResponse.json({error:'访问凭证无效或已撤销'}, {status:401});
    if (result.identity.role !== 'operator') return NextResponse.json({error:'Agent 凭证仅用于 MCP；控制台需要操作员凭证'}, {status:403});
    const response = NextResponse.json(result);
    response.cookies.set(cookie, token, {httpOnly:true, sameSite:'strict', secure:request.nextUrl.protocol === 'https:', path:'/', maxAge:8*3600});
    response.headers.set('Cache-Control', 'no-store');
    return response;
  } catch (error) {
    if(error instanceof SyntaxError) return NextResponse.json({error:'凭证格式无效'},{status:400});
    return NextResponse.json({error:'身份服务未连接或凭证格式无效'}, {status:503});
  }
}
export async function DELETE(request: NextRequest) {
  if (!sameOrigin(request)) return NextResponse.json({error:'请求来源不匹配'}, {status:403});
  const response = NextResponse.json({ok:true});
  response.cookies.set(cookie, '', {httpOnly:true, sameSite:'strict', path:'/', maxAge:0});
  return response;
}
