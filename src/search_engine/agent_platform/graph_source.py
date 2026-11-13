"""Read graph facts from the existing business service, without seeding or writing it."""
from .common import text

class GraphSource:
    def __init__(self,factory=None):
        self.factory=factory or self.connect

    @staticmethod
    def connect():
        try:
            from ..index_tab.janusgraph_backend import JanusGraphBackend, JanusGraphBackendError
        except ImportError:
            raise ValueError('图谱来源未就绪，请检查 Gremlin 服务及原图谱依赖') from None
        try:
            return JanusGraphBackend(connect_timeout_sec=3,connect_retries=1)
        except JanusGraphBackendError:
            raise ValueError('图谱来源未就绪，请检查 Gremlin 服务及原图谱依赖') from None

    def read(self,entity):
        entity=text(entity,'图谱实体',200)
        service=self.factory()
        try:
            relations=service.get_entity_relations(entity)
            facts=[]
            for direction in ('outgoing','incoming'):
                for row in relations.get(direction,[])[:20]:
                    subject=entity if direction=='outgoing' else row['source']
                    target=row['target'] if direction=='outgoing' else entity
                    facts.append({'subject':subject,'predicate':row['predicate'],'object':target,
                                  'document_id':row.get('doc_id','')})
            if not facts:raise ValueError('图谱未返回该实体的关系；未创建空白资料')
            return {'schema':'graph-facts-v1','entity':entity,'facts':facts,'provider':'legacy-graph'}
        finally:service.close()
