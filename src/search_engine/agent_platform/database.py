import os
import time
from pathlib import Path

from sqlalchemy import JSON, Float, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Owned:
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    project: Mapped[str] = mapped_column(String(80), index=True)
    actor: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class Position(Owned, Base):
    __tablename__ = 'positions'
    name: Mapped[str] = mapped_column(String(120))
    duties: Mapped[str] = mapped_column(Text)


class Agent(Owned, Base):
    __tablename__ = 'agents'
    position_id: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(120))
    current_release: Mapped[str | None] = mapped_column(String(32), nullable=True)


class ModelService(Owned, Base):
    __tablename__ = 'model_services'
    name: Mapped[str] = mapped_column(String(120))
    model: Mapped[str] = mapped_column(String(200))
    base_url: Mapped[str] = mapped_column(String(2000))
    context_window: Mapped[int] = mapped_column(Integer)
    max_tokens: Mapped[int] = mapped_column(Integer)
    has_credential: Mapped[bool] = mapped_column(default=False)


class Release(Owned, Base):
    __tablename__ = 'agent_releases'
    agent_id: Mapped[str] = mapped_column(String(32), index=True)
    version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24), default='candidate')
    digest: Mapped[str] = mapped_column(String(64))
    manifest: Mapped[dict] = mapped_column(JSON)
    adopted_run: Mapped[str | None] = mapped_column(String(32), nullable=True)
    __table_args__ = (UniqueConstraint('agent_id', 'version'),)


class Run(Owned, Base):
    __tablename__ = 'agent_runs'
    agent_id: Mapped[str] = mapped_column(String(32), index=True)
    release_id: Mapped[str] = mapped_column(String(32))
    request_id: Mapped[str] = mapped_column(String(64))
    task: Mapped[str] = mapped_column(Text)
    execution_identity: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(24), default='queued')
    result: Mapped[str] = mapped_column(Text, default='')
    error: Mapped[str] = mapped_column(Text, default='')
    native_session: Mapped[str] = mapped_column(Text, default='')
    finished_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    __table_args__ = (UniqueConstraint('project', 'request_id'),)


class Event(Base):
    __tablename__ = 'agent_events'
    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    body: Mapped[dict] = mapped_column(JSON)


def connect(directory, url=None):
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    url_file = directory / 'database-url.txt'
    url = url or os.environ.get('WORKBENCH_DATABASE_URL') or (
        url_file.read_text().strip() if url_file.exists() else
        'sqlite:///' + str(directory / 'agent-platform-v1.sqlite3'))
    engine = create_engine(url, connect_args={'check_same_thread': False} if url.startswith('sqlite:') else {},
                           pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return engine, sessionmaker(engine, expire_on_commit=False)
