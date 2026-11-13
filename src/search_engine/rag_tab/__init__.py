"""Business retrieval can be imported without the Gradio interface."""
from .rag_service import RAGService

def __getattr__(name):
    if name=='build_rag_tab':
        from .rag_tab import build_rag_tab
        return build_rag_tab
    raise AttributeError(name)

__all__=['build_rag_tab','RAGService']
