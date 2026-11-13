from pathlib import Path
from .database import connect
from .catalog import Catalog
from .pi_runtime import PiRuntime
from .runs import Runs


class AgentPlatform:
    def __init__(self, directory, context, access, database_url=None, background=True):
        self.directory = Path(directory)
        self.engine, self.session = connect(directory, database_url)
        self.catalog = Catalog(self.session, directory, context)
        self.runtime = PiRuntime(directory)
        self.runs = Runs(self.session, self.runtime, access, background)

    def overview(self):
        return {**self.catalog.listing(), 'harnesses': [{'kind': 'pi', 'version': '1.5.2',
                'installed': self.runtime.available(), 'tools': ['read_enterprise_context']}],
                'storage': 'postgresql' if self.engine.dialect.name == 'postgresql' else 'sqlite',
                'scope': '单机首版；原生 Pi，预配上下文只读工具，OpenAI Chat Completions 模型服务'}

    def close(self):
        self.runs.close()
        self.engine.dispose()
