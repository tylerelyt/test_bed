"""Conversation history is private to a project, identity and session."""
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from .workbench_access import history_key


class HistoryStore:
    def __init__(self, directory=None):
        directory = Path(directory or os.environ.get('WORKBENCH_STATE_DIR') or Path(__file__).resolve().parents[2] / '.workbench')
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = directory / 'conversation-history-v1.sqlite3'
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS history (project TEXT, actor TEXT, session TEXT, at REAL, body TEXT)')
            db.execute('CREATE INDEX IF NOT EXISTS history_scope ON history(project,actor,session)')
        self.path.chmod(0o600)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def read(self):
        with self.db() as db:
            rows = db.execute('SELECT body FROM history WHERE project=? AND actor=? AND session=? ORDER BY rowid', history_key()).fetchall()
        return [json.loads(row[0]) for row in rows]

    def append(self, record):
        if not isinstance(record, dict):
            raise ValueError('历史记录须为对象')
        body = json.dumps(record, ensure_ascii=False, allow_nan=False)
        if len(body) > 2_000_000:
            raise ValueError('单轮历史过大')
        with self.db() as db:
            db.execute('INSERT INTO history VALUES (?,?,?,?,?)', (*history_key(), time.time(), body))

    def clear(self):
        with self.db() as db:
            db.execute('DELETE FROM history WHERE project=? AND actor=? AND session=?', history_key())
