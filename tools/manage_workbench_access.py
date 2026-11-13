"""Run locally to issue or revoke individual workbench credentials."""
import argparse
import hashlib
import json
import secrets
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.search_engine.workbench_access import AccessPolicy, private_write


def main():
    parser = argparse.ArgumentParser(description='创建、轮换或撤销工作台身份；凭证仅写入本机私有文件')
    parser.add_argument('action', choices=['issue', 'disable'])
    parser.add_argument('--id', required=True)
    parser.add_argument('--project', default='local')
    parser.add_argument('--role', choices=['operator', 'agent'], default='agent')
    parser.add_argument('--bundle', action='append', default=[])
    parser.add_argument('--business-access', action='store_true', help='明确允许访问原共享业务模块，仅供可信操作员')
    args = parser.parse_args()
    policy = AccessPolicy()
    entries = policy.entries()
    old = next((item for item in entries if item['id'] == args.id), None)
    if args.action == 'disable':
        if old is None: parser.error('身份不存在')
        old['enabled'] = False
        private_write(policy.path, json.dumps({'version': 1, 'principals': entries}, ensure_ascii=False, indent=2))
        print('身份已撤销；后续请求和周期执行将重新核对授权。')
        return
    if args.role == 'agent' and args.business_access: parser.error('Agent 不能获得原业务模块操作权限')
    token = secrets.token_urlsafe(32)
    entry = {'id': args.id, 'project': args.project, 'role': args.role,
             'token_sha256': hashlib.sha256(token.encode()).hexdigest(), 'bundles': args.bundle,
             'business_access': args.business_access, 'enabled': True}
    # Validate identifiers using the same parser before changing the real file.
    import re
    if not all(re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', value) for value in (args.id, args.project)):
        parser.error('身份与项目标识只能包含字母、数字、点、下划线和连字符，最多80字符')
    entries = [item for item in entries if item['id'] != args.id] + [entry]
    credentials = policy.directory / 'credentials'
    credentials.mkdir(exist_ok=True, mode=0o700)
    output = credentials / (args.id + '.txt')
    private_write(output, token + '\n')
    private_write(policy.path, json.dumps({'version': 1, 'principals': entries}, ensure_ascii=False, indent=2))
    print('身份已创建或轮换；凭证文件：' + str(output))


if __name__ == '__main__':
    main()
