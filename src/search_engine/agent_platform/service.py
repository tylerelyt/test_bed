from pathlib import Path
from .database import connect
from .catalog import Catalog
from .pi_runtime import PiRuntime
from .runs import Runs
from .models import ModelSupply
from .jobs import Jobs
from .environment import capabilities
from .business import BusinessOperations
from .knowledge import KnowledgeSupply
from .model_operations import ModelOperations
from .validation import SupplyValidation


class AgentPlatform:
    def __init__(self, directory, context, access, database_url=None, background=True, legacy_index=None):
        self.directory = Path(directory)
        self.engine, self.session = connect(directory, database_url)
        self.knowledge=KnowledgeSupply(self.session,directory,context,legacy_index=legacy_index)
        self.catalog = Catalog(self.session, directory, context, self.knowledge)
        self.validation=SupplyValidation(self.session,directory,self.knowledge)
        self.runtime = PiRuntime(directory)
        self.runs = Runs(self.session, self.runtime, access, background)
        self.context = context
        self.models = ModelSupply(self.session, directory)
        self.models.operations=ModelOperations(self.session,directory,self.models,self.catalog.models)
        self.business = BusinessOperations(self.session,context,self.models)
        self.jobs = Jobs(self.session, directory, access, self.knowledge, self.models)
        if background:
            self.jobs.start_worker()

    def overview(self):
        return {**self.catalog.listing(), 'harnesses': [{'kind': 'pi', 'version': '1.5.2',
                'installed': self.runtime.available(), 'tools': ['read_enterprise_context', 'search_enterprise_knowledge']}],
                'storage': 'postgresql' if self.engine.dialect.name == 'postgresql' else 'sqlite',
                'scope': '单机首版；原生 Pi，预配上下文与固定索引只读工具，OpenAI Chat Completions 模型服务'}

    def environment(self):
        return {**capabilities(self.directory, self.engine.dialect.name),
                'harnesses': [{'kind': 'pi', 'version': '1.5.2', 'installed': self.runtime.available()}]}

    def close(self):
        self.runs.close()
        self.jobs.close()
        self.engine.dispose()
