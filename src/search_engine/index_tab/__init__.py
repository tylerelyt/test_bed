# Load the UI only when requested; offline retrieval does not need Gradio.
def __getattr__(name):
    if name in {'build_index_tab', 'show_index_stats', 'check_index_quality', 'view_inverted_index'}:
        from . import index_tab
        return getattr(index_tab, name)
    raise AttributeError(name)

from .offline_index import InvertedIndex, create_sample_documents, build_index_from_documents
from .index_service import IndexServiceInterface, InvertedIndexService, get_index_service, reset_index_service

__all__ = [
    'build_index_tab', 'show_index_stats', 'check_index_quality', 'view_inverted_index',
    'InvertedIndex', 'create_sample_documents', 'build_index_from_documents',
    'IndexServiceInterface', 'InvertedIndexService', 'get_index_service', 'reset_index_service'
] 