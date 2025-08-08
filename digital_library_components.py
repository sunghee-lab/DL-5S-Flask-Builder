from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from datetime import datetime, timezone
import uuid

@dataclass
class Stream:
    kind: str  # 'text', 'image', 'audio', ...
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ''
    content: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"Stream(kind={self.kind}, id={self.id})"

@dataclass
class Structure:
    kind: str = 'graph'
    nodes: Set[str] = field(default_factory=set)
    edges: Set[Tuple[str, str]] = field(default_factory=set)
    labels: Dict[Any, Any] = field(default_factory=dict)

    def add_node(self, node: str, label: Optional[Any] = None):
        self.nodes.add(node)
        if label is not None:
            self.labels[node] = label

    def add_edge(self, u: str, v: str, label: Optional[Any] = None):
        self.edges.add((u, v))
        if label is not None:
            self.labels[(u, v)] = label

    def neighbors(self, node: str):
        return [v for (u, v) in self.edges if u == node]

    def __repr__(self):
        return f"Structure(kind={self.kind}, nodes={len(self.nodes)}, edges={len(self.edges)})"

@dataclass
class Space:
    name: str  # 'vector', 'semantic', '2D', '3D', ...
    dimensions: List[str] = field(default_factory=list)

    def __repr__(self):
        return f"Space(name={self.name}, dims={self.dimensions})"

@dataclass
class Scenario:
    name: str
    steps: List[Callable] = field(default_factory=list)

    def run(self, middleware, ctx: Optional[Dict] = None):
        ctx = ctx or {}
        for step in self.steps:
            step(middleware, ctx)
        return ctx

@dataclass
class Society:
    actors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    roles: Dict[str, Set[str]] = field(default_factory=dict)

    def add_actor(self, actor_id: str, attrs: Optional[Dict[str, Any]] = None):
        self.actors[actor_id] = attrs or {}

    def assign_role(self, actor_id: str, role: str):
        self.roles.setdefault(role, set()).add(actor_id)

    def __repr__(self):
        return f"Society(actors={len(self.actors)})"

class DigitalLibrary:
    def __init__(self, stream: Stream, structure: Structure, space: Space, scenario: Scenario, society: Society):
        self.stream = stream
        self.structure = structure
        self.space = space
        self.scenario = scenario
        self.society = society
        self.created_at = datetime.now(timezone.utc)
        self.indexing_service = None
        self.retrieval_service = None

    def __repr__(self):
        return (f"DigitalLibrary(stream={self.stream.kind}, structure={self.structure.kind}, "
                f"space={self.space.name}, scenario={self.scenario.name})")
