# bpmn_utils.py

from dataclasses import dataclass, field
import logging
logger = logging.getLogger(__name__)

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
import pm4py
from pm4py.objects.bpmn.obj import BPMN


@dataclass(frozen=True)  # Making it immutable and hashable
class BPMNNode:
    """Wrapper class for BPMN nodes to provide consistent interface"""
    id: str
    type: str
    name: str
    incoming: tuple  # Changed from List to tuple to make it hashable
    outgoing: tuple  # Changed from List to tuple to make it hashable
    original_node: Any = field(hash=False)  # Exclude from hash calculation

    def __hash__(self):
        return hash((self.id, self.type, self.name))

    def __eq__(self, other):
        if not isinstance(other, BPMNNode):
            return False
        return self.id == other.id

class BPMNGraphUtils:
    """Utility class to handle BPMN graph operations safely"""

    def __init__(self, bpmn_graph):
        self.bpmn_graph = bpmn_graph
        self.nodes = {}  # Dictionary to store wrapped nodes
        self.node_map = {}  # Dictionary to store original nodes
        self._initialize_nodes()

    def _initialize_nodes(self):
        """Initialize node mappings"""
        try:
            nodes = (self.bpmn_graph.get_nodes() if hasattr(self.bpmn_graph, 'get_nodes')
                     else getattr(self.bpmn_graph, 'nodes', []))

            for node in nodes:
                node_id = self._get_node_id(node)
                node_type = self._get_node_type(node)
                node_name = self._get_node_name(node)
                incoming = tuple(self._get_incoming_ids(node))  # Convert to tuple
                outgoing = tuple(self._get_outgoing_ids(node))  # Convert to tuple

                wrapped_node = BPMNNode(
                    id=node_id,
                    type=node_type,
                    name=node_name,
                    incoming=incoming,
                    outgoing=outgoing,
                    original_node=node
                )

                self.nodes[node_id] = wrapped_node
                self.node_map[node_id] = node

        except Exception as e:
            logger.error(f"Error initializing BPMN nodes: {str(e)}")
            self.nodes = {}
            self.node_map = {}

    def get_upstream_nodes(self, node_id: str) -> Set[str]:
        """Get all nodes upstream from given node"""
        upstream = set()  # Store node IDs
        visited = set()
        node = self.get_node_by_id(node_id)

        if not node:
            return set()

        def dfs(current_node: BPMNNode):
            if current_node.id in visited:
                return
            visited.add(current_node.id)

            for incoming_id in current_node.incoming:
                source_node = self.get_node_by_id(incoming_id)
                if source_node:
                    upstream.add(incoming_id)
                    dfs(source_node)

        dfs(node)
        return upstream

    def get_downstream_nodes(self, node_id: str) -> Set[str]:
        """Get all nodes downstream from given node"""
        downstream = set()  # Store node IDs
        visited = set()
        node = self.get_node_by_id(node_id)

        if not node:
            return set()

        def dfs(current_node: BPMNNode):
            if current_node.id in visited:
                return
            visited.add(current_node.id)

            for outgoing_id in current_node.outgoing:
                target_node = self.get_node_by_id(outgoing_id)
                if target_node:
                    downstream.add(outgoing_id)
                    dfs(target_node)

        dfs(node)
        return downstream

    def get_all_paths(self) -> List[List[str]]:
        """Get all possible paths through the BPMN graph"""
        paths = []
        start_nodes = self.get_start_nodes()

        def dfs(node: BPMNNode, current_path: List[str], visited: Set[str]):
            if 'end_event' in node.type.lower():
                paths.append(current_path + [node.id])
                return

            for target_id in node.outgoing:
                if target_id not in visited:
                    target_node = self.get_node_by_id(target_id)
                    if target_node:
                        new_visited = visited | {target_id}
                        dfs(target_node, current_path + [node.id], new_visited)

        for start_node in start_nodes:
            dfs(start_node, [], {start_node.id})

        return paths

    def get_node_by_id(self, node_id: str) -> Optional[BPMNNode]:
        """Get wrapped node by ID"""
        return self.nodes.get(str(node_id))

    def get_original_node(self, node_id: str) -> Optional[Any]:
        """Get original node by ID"""
        return self.node_map.get(str(node_id))

    def get_all_nodes(self) -> List[BPMNNode]:
        """Get all wrapped nodes"""
        return list(self.nodes.values())

    def get_nodes_by_type(self, node_type: str) -> List[BPMNNode]:
        """Get all nodes of a specific type"""
        return [node for node in self.nodes.values() if node.type.lower() == node_type.lower()]

    def get_start_nodes(self) -> List[BPMNNode]:
        """Get all start event nodes"""
        return self.get_nodes_by_type('start_event')

    def get_end_nodes(self) -> List[BPMNNode]:
        """Get all end event nodes"""
        return self.get_nodes_by_type('end_event')

    def is_gateway(self, node_id: str) -> bool:
        """Check if node is a gateway"""
        node = self.get_node_by_id(node_id)
        return node and ('gateway' in node.type.lower())

    def is_task(self, node_id: str) -> bool:
        """Check if node is a task"""
        node = self.get_node_by_id(node_id)
        return node and ('task' in node.type.lower())

    def _get_node_id(self, node) -> str:
        """Safely get node ID"""
        if hasattr(node, 'get_id'):
            return str(node.get_id())
        elif hasattr(node, 'id'):
            return str(node.id)
        return str(id(node))

    def _get_node_type(self, node) -> str:
        """Safely get node type"""
        if hasattr(node, 'get_type'):
            return node.get_type()
        elif isinstance(node, BPMN.Task):
            return 'task'
        elif isinstance(node, BPMN.ExclusiveGateway):
            return 'exclusive_gateway'
        elif isinstance(node, BPMN.ParallelGateway):
            return 'parallel_gateway'
        elif isinstance(node, BPMN.StartEvent):
            return 'start_event'
        elif isinstance(node, BPMN.EndEvent):
            return 'end_event'
        return node.__class__.__name__.lower()

    def _get_node_name(self, node) -> str:
        """Safely get node name"""
        if hasattr(node, 'get_name'):
            return node.get_name() or ''
        elif hasattr(node, 'name'):
            return str(node.name) or ''
        return ''

    def _get_incoming_ids(self, node) -> List[str]:
        """Safely get incoming arc IDs"""
        incoming = []
        try:
            if hasattr(node, 'get_incoming'):
                arcs = node.get_incoming()
            else:
                arcs = getattr(node, 'in_arcs', [])

            for arc in arcs:
                source = arc.source if hasattr(arc, 'source') else arc.get_source()
                incoming.append(self._get_node_id(source))
        except Exception:
            pass
        return incoming

    def _get_outgoing_ids(self, node) -> List[str]:
        """Safely get outgoing arc IDs"""
        outgoing = []
        try:
            if hasattr(node, 'get_outgoing'):
                arcs = node.get_outgoing()
            else:
                arcs = getattr(node, 'out_arcs', [])

            for arc in arcs:
                target = arc.target if hasattr(arc, 'target') else arc.get_target()
                outgoing.append(self._get_node_id(target))
        except Exception:
            pass
        return outgoing

    def get_path_length(self, path: List[str]) -> int:
        """Get the length of a path"""
        return len(path)

    def get_path_complexity(self, path: List[str]) -> float:
        """Calculate complexity of a path based on node types"""
        complexity = 0.0
        for node_id in path:
            node = self.get_node_by_id(node_id)
            if node:
                if self.is_gateway(node_id):
                    complexity += 1.0
                elif self.is_task(node_id):
                    complexity += 0.5
        return complexity