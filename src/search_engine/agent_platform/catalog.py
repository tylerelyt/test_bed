"""Composition façade; each domain service owns its writes."""
from sqlalchemy import select
from ..workbench_access import identity
from .common import public
from .database import Position, Agent, ModelService, Release, Run
from .positions import Positions
from .model_services import ModelServices
from .releases import Releases


class Catalog:
    def __init__(self, session, directory, context, knowledge=None):
        self.session = session
        self.positions = Positions(session)
        self.models = ModelServices(session, directory)
        self.releases = Releases(session, context, knowledge)
        self.position, self.agent = self.positions.position, self.positions.agent
        self.model = self.models.model
        self.release, self.publish = self.releases.release, self.releases.publish

    def listing(self):
        with self.session() as db:
            return {name: [public(x) for x in db.scalars(select(cls).where(cls.project == identity().project)
                    .order_by(cls.created_at.desc()).limit(200))]
                    for name, cls in [('positions', Position), ('agents', Agent), ('models', ModelService),
                                      ('releases', Release), ('runs', Run)]}
