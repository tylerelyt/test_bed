from ..workbench_access import require_operator
from .common import get, owned, public, text
from .database import Agent, Position


class Positions:
    def __init__(self, session):
        self.session = session

    def position(self, name, duties):
        require_operator()
        with self.session.begin() as db:
            item = Position(**owned(), name=text(name, '岗位名称'), duties=text(duties, '岗位职责', 12000))
            db.add(item)
            db.flush()
            return public(item)

    def agent(self, position_id, name):
        require_operator()
        with self.session.begin() as db:
            position = get(db, Position, position_id)
            item = Agent(**owned(), position_id=position.id, name=text(name, 'Agent 名称'))
            db.add(item)
            db.flush()
            return public(item)
