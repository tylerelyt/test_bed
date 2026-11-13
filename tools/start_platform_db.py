"""Project-owned PostgreSQL data directory; does not register a system service."""
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.search_engine.workbench_access import private_write


def main():
    import psycopg
    root = Path(__file__).resolve().parents[1]
    state = Path(os.environ.get('WORKBENCH_STATE_DIR') or root / '.workbench')
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    binary = os.environ.get('WORKBENCH_POSTGRES_BIN')
    if not binary:
        found = shutil.which('pg_ctl')
        binary = str(Path(found).parent) if found else '/opt/homebrew/opt/postgresql@17/bin'
    binary = Path(binary)
    if not (binary / 'pg_ctl').is_file():
        raise SystemExit('请安装 PostgreSQL 17，并通过 WORKBENCH_POSTGRES_BIN 指定其 bin 目录')
    data, password_file = state / 'postgres', state / 'postgres-password.txt'
    if not password_file.exists():
        private_write(password_file, secrets.token_hex(32))
    if not (data / 'PG_VERSION').exists():
        subprocess.run([str(binary/'initdb'), '-D', str(data), '--username=workbench',
            '--auth-local=trust', '--auth-host=scram-sha-256', '--pwfile='+str(password_file),
            '--encoding=UTF8', '--locale=C'], check=True, stdout=subprocess.DEVNULL)
    port = int(os.environ.get('WORKBENCH_POSTGRES_PORT', '55432'))
    running = subprocess.run([str(binary/'pg_ctl'), '-D', str(data), 'status'], capture_output=True).returncode == 0
    if not running:
        sockets = state/'postgres-sockets'
        sockets.mkdir(mode=0o700, exist_ok=True)
        subprocess.run([str(binary/'pg_ctl'), '-D', str(data), '-l', str(state/'postgres.log'),
            '-o', f'-h 127.0.0.1 -p {port} -k {sockets}', '-w', 'start'], check=True, stdout=subprocess.DEVNULL)
    password = password_file.read_text().strip()
    with psycopg.connect(host='127.0.0.1', port=port, user='workbench', password=password,
                         dbname='postgres', autocommit=True) as db:
        if not db.execute("SELECT 1 FROM pg_database WHERE datname='workbench'").fetchone():
            db.execute('CREATE DATABASE workbench')
    private_write(state/'database-url.txt', f'postgresql+psycopg://workbench:{password}@127.0.0.1:{port}/workbench')
    print('项目 PostgreSQL 已就绪；连接配置保存在 .workbench/database-url.txt，未注册开机服务')


if __name__ == '__main__':
    main()
