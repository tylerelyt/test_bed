"""Adapter to the existing lexical retrieval implementation; no UI or global service."""
from pathlib import Path
import hashlib
import importlib.metadata


class LexicalIndex:
    @staticmethod
    def fingerprint():
        source=Path(__file__).resolve().parents[1]/'index_tab/offline_index.py'
        return hashlib.sha256(Path(__file__).read_bytes()+source.read_bytes()+
                              importlib.metadata.version('jieba').encode()).hexdigest()

    @staticmethod
    def implementation():
        from ..index_tab.offline_index import InvertedIndex
        return InvertedIndex

    def build(self, documents, path):
        index = self.implementation()()
        for document in documents:
            index.add_document(document['id'], document['content'])
        index.save_to_file(str(path))

    def load(self, path):
        index = self.implementation()()
        index.load_from_file(str(path))
        return index

    def documents(self, path):
        if not Path(path).is_file():
            raise ValueError('原文档索引不存在，请先在原索引功能中准备数据')
        return dict(self.load(path).documents)

    def search(self, path, sources, query, limit):
        index = self.load(path)
        metadata = {source['id']: source for source in sources}
        return [{**metadata[doc_id], 'score': score, 'content': index.get_document(doc_id)}
                for doc_id, score, _ in index.search(query, top_k=limit)]
