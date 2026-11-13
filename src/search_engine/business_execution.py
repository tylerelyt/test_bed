"""Request-local business dependencies; no process-wide credential/environment mutation."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

_current=ContextVar('business_execution',default=None)

@contextmanager
def execution_scope(run, directory, records):
    marker=_current.set({'run':run,'directory':Path(directory),'records':records,'processes':[]})
    try:yield _current.get()
    finally:_current.reset(marker)

def current_execution():return _current.get()

def OpenAI(*args,**kwargs):
    # Resolve at call time: long-lived NER/vision clients must not capture a
    # different run's model or credentials when their service is initialized.
    from openai import OpenAI as SDK
    def base():return SDK(*args,**kwargs)
    class Completions:
        def create(self,**parameters):
            scope=_current.get()
            config=scope['run'].get('model_service') if scope else None
            if not config:return base().chat.completions.create(**parameters)
            from .workbench_access import require_business
            require_business()
            key_file=scope['directory']/'model-credentials'/(config['id']+'.txt')
            key=key_file.read_text().strip() if config['has_credential'] else 'workbench-local'
            parameters['model']=config['model']
            parameters['max_tokens']=min(parameters.get('max_tokens',config['max_tokens']),config['max_tokens'])
            return SDK(api_key=key,base_url=config['base_url']).chat.completions.create(**parameters)
    class Bound:
        chat=SimpleNamespace(completions=Completions())
        def __getattr__(self,name):return getattr(base(),name)
    return Bound()
