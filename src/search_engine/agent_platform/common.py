import uuid
from ..workbench_access import identity

def text(value, label, maximum=120):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(label + '为空或长度超限')
    return value.strip()


def owned():
    who = identity()
    return {'id': uuid.uuid4().hex, 'project': who.project, 'actor': who.id}


def get(db, cls, item_id):
    item = db.get(cls, item_id)
    if item is None or item.project != identity().project:
        raise ValueError('记录不存在或不属于当前项目')
    return item


def public(item):
    return {column.name: getattr(item, column.name) for column in item.__table__.columns
            if column.name != 'execution_identity'}
