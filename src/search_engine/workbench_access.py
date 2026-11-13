"""Authenticated identities and project scopes shared by HTTP, jobs and MCP."""
import base64
import contextvars
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class AccessDenied(PermissionError):
    def __init__(self, message='没有执行此操作的权限', status=403):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Identity:
    id: str
    project: str
    role: str
    fingerprint: str = ''
    bundles: tuple = ()
    business_access: bool = False

    def public(self):
        return {'id': self.id, 'project': self.project, 'role': self.role,
                'business_access': self.business_access}

    def claim(self):
        return {**self.public(), 'fingerprint': self.fingerprint}


_identity = contextvars.ContextVar('workbench_identity', default=None)
_policy = contextvars.ContextVar('workbench_access_policy', default=None)
_session = contextvars.ContextVar('workbench_history_session', default=None)
LOCAL = Identity('local-internal', 'local', 'operator', business_access=True)


def identity():
    return _identity.get() or LOCAL


def project():
    return identity().project


def require_operator():
    if identity().role != 'operator':
        raise AccessDenied('此操作需要项目操作员身份')


def require_business():
    require_operator()
    if not identity().business_access:
        raise AccessDenied('此身份未获准执行原业务模块；仍可维护所属项目的上下文')


def visible(item):
    return item.get('project', 'local') == project()


def check(item):
    if not visible(item):
        raise AccessDenied('无权访问该项目的记录')
    return item


def session_key(raw):
    who = identity()
    if who is LOCAL:
        return raw
    return hashlib.sha256((who.project + '\0' + who.id).encode()).hexdigest()[:24] + ':' + raw


@contextmanager
def history_session(raw):
    if not isinstance(raw, str) or not raw or len(raw) > 160:
        raise ValueError('会话标识无效')
    token = _session.set(raw)
    try:
        yield
    finally:
        _session.reset(token)


def history_key():
    raw = _session.get()
    if not raw:
        raise AccessDenied('读取或清理历史需要明确的会话')
    who = identity()
    return who.project, who.id, raw


def private_write(path, text):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as target:
            target.write(text)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def private_create(path, text):
    """Atomically install a complete private file without replacing a winner."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            output.write(text)
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
            return True
        except FileExistsError:
            return False
    finally:
        Path(temporary).unlink(missing_ok=True)


class AccessPolicy:
    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get('WORKBENCH_STATE_DIR') or Path(__file__).resolve().parents[2] / '.workbench')
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        explicit = os.environ.get('WORKBENCH_ACCESS_FILE')
        self.path = Path(explicit) if explicit else self.directory / 'access.json'
        if not self.path.exists():
            if explicit:
                raise ValueError('WORKBENCH_ACCESS_FILE 不存在，拒绝以无认证模式启动')
            self.bootstrap()
        keyfile = self.directory / 'access-signing.key'
        private_create(keyfile, secrets.token_hex(32))
        self.key = keyfile.read_bytes()
        if not re.fullmatch(rb'[a-f0-9]{64}', self.key):
            raise AccessDenied('内部签名密钥无效，访问已关闭', 503)
        self.entries()

    def bootstrap(self):
        token = secrets.token_urlsafe(32)
        entry = {'id': 'local-owner', 'project': 'local', 'role': 'operator',
                 'token_sha256': hashlib.sha256(token.encode()).hexdigest(), 'business_access': True}
        if private_create(self.path, json.dumps({'version': 1, 'principals': [entry]}, ensure_ascii=False, indent=2)):
            private_write(self.directory / 'bootstrap-access.txt', token + '\n')

    def entries(self):
        try:
            data = json.loads(self.path.read_text())
            values = data['principals']
            if not isinstance(values, list) or not values:
                raise ValueError()
            ids, fingerprints = set(), set()
            for item in values:
                if (not isinstance(item, dict) or item.get('id') in ids or item.get('token_sha256') in fingerprints or
                    not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', item.get('id', '')) or
                    not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', item.get('project', '')) or
                    item.get('role') not in ('operator', 'agent') or
                    not re.fullmatch(r'[a-f0-9]{64}', item.get('token_sha256', '')) or
                    not isinstance(item.get('bundles', []), list) or
                    not all(isinstance(value, str) for value in item.get('bundles', [])) or
                    not isinstance(item.get('business_access', False), bool) or
                    not isinstance(item.get('enabled', True), bool) or
                    (item.get('role') == 'agent' and item.get('business_access', False))):
                    raise ValueError()
                ids.add(item['id'])
                fingerprints.add(item['token_sha256'])
            return values
        except (OSError, ValueError, TypeError, KeyError):
            raise AccessDenied('身份配置无效，访问已关闭', 503)

    def from_entry(self, entry):
        return Identity(entry['id'], entry['project'], entry['role'], entry['token_sha256'],
                        tuple(entry.get('bundles', [])), entry.get('business_access', False))

    def revalidate(self, principal):
        entry = next((item for item in self.entries() if item['id'] == principal.id and
                      item.get('enabled', True) and item['project'] == principal.project and item['role'] == principal.role
                      and hmac.compare_digest(item['token_sha256'], principal.fingerprint)), None)
        if entry is None:
            raise AccessDenied('访问凭证已撤销，请重新登录', 401)
        return self.from_entry(entry)

    def authenticate(self, authorization, *, delegation=False):
        token = authorization[7:] if authorization and authorization.startswith('Bearer ') else ''
        if delegation and token.startswith('wb1.'):
            try:
                _, payload, signature = token.split('.')
                expected = hmac.new(self.key, payload.encode(), hashlib.sha256).hexdigest()
                claims = json.loads(base64.urlsafe_b64decode(payload))
                if not hmac.compare_digest(signature, expected) or claims['expires'] < time.time():
                    raise ValueError()
                principal = Identity(claims['id'], claims['project'], claims['role'], claims['fingerprint'])
                return self.revalidate(principal)
            except (ValueError, TypeError, KeyError):
                raise AccessDenied('内部委托凭证无效', 401)
        fingerprint = hashlib.sha256(token.encode()).hexdigest()
        entry = next((item for item in self.entries() if item.get('enabled', True) and
                      hmac.compare_digest(item['token_sha256'], fingerprint)), None)
        if not token or entry is None:
            raise AccessDenied('需要有效的访问凭证', 401)
        return self.from_entry(entry)

    def delegated_headers(self):
        principal = self.revalidate(identity())
        payload = base64.urlsafe_b64encode(json.dumps({**principal.claim(), 'expires': time.time() + 60}).encode()).decode()
        signature = hmac.new(self.key, payload.encode(), hashlib.sha256).hexdigest()
        headers = {'Authorization': 'Bearer wb1.' + payload + '.' + signature}
        if _session.get():
            headers['X-Workbench-Session'] = _session.get()
        return headers

    @contextmanager
    def bind(self, principal):
        a, b = _identity.set(principal), _policy.set(self)
        try:
            yield
        finally:
            _policy.reset(b)
            _identity.reset(a)

    def resume(self, claims):
        return self.bind(self.revalidate(Identity(claims['id'], claims['project'], claims['role'], claims['fingerprint'])))


def client_headers():
    policy = _policy.get()
    return policy.delegated_headers() if policy else {}


def mcp_auth(policy):
    from fastmcp.server.auth import TokenVerifier, AccessToken

    class Verifier(TokenVerifier):
        async def verify_token(self, token):
            try:
                who = policy.authenticate('Bearer ' + token, delegation=True)
                return AccessToken(token=token, client_id=who.id, scopes=['workbench'])
            except AccessDenied:
                return None
    return Verifier()


def configure_mcp(mcp, policy):
    from fastmcp.server.middleware import Middleware
    from fastmcp.server.dependencies import get_http_headers

    class AccessMiddleware(Middleware):
        async def on_request(self, context, call_next):
            headers = get_http_headers()
            who = policy.authenticate(headers.get('authorization'), delegation=True)
            with policy.bind(who):
                if context.method == 'tools/call' and getattr(context.message, 'name', None) not in (
                    'discover_context', 'load_context', 'write_context_notebook'):
                    require_business()
                if who.role == 'agent':
                    if context.method == 'tools/call' and getattr(context.message, 'name', None) not in (
                        'discover_context', 'load_context', 'write_context_notebook'):
                        raise AccessDenied('Agent 仅可调用获准的上下文供应工具')
                    if context.method == 'resources/read' and not str(getattr(context.message, 'uri', '')).startswith('context://'):
                        raise AccessDenied('Agent 仅可读取获准的上下文资源')
                raw = headers.get('x-workbench-session')
                if not raw and context.fastmcp_context:
                    raw = context.fastmcp_context.session_id
                with history_session(raw or 'mcp-initialization'):
                    return await call_next(context)
    mcp.add_middleware(AccessMiddleware())
