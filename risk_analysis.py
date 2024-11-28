# risk_analysis.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import pm4py
import numpy as np
from collections import defaultdict
from pm4py.objects.bpmn.obj import BPMN


@dataclass
class FailureMode:
    node_id: str
    node_type: str
    description: str
    historical_frequency: float = 0.0
    upstream_activities: List[str] = None
    downstream_activities: List[str] = None


class ProcessRiskAnalyzer:
    def __init__(self, event_log, bpmn_graph):
        self.event_log = event_log
        self.bpmn_graph = bpmn_graph
        self.failure_modes = []
        self.activity_stats = self._compute_activity_statistics()

    def _compute_activity_statistics(self) -> Dict:
        """Compute statistics about activities from event log"""
        stats = defaultdict(lambda: {
            'frequency': 0,
            'duration': [],
            'variations': set(),
            'completion_rate': 0
        })

        for trace in self.event_log:
            previous_activity = None
            for event in trace:
                activity = event['concept:name']
                stats[activity]['frequency'] += 1
                if previous_activity:
                    stats[activity]['variations'].add(previous_activity)
                previous_activity = activity

        return dict(stats)

    def analyze_bpmn_graph(self):
        """Analyze BPMN graph for potential failure modes"""
        print("invoked analyze_bpmn_graph...")

        for node in self.bpmn_graph.get_nodes():
            # Get incoming and outgoing arcs using the correct BPMN methods
            incoming = node.get_in_arcs()
            outgoing = node.get_out_arcs()

            # Get node type using isinstance checks
            if isinstance(node, BPMN.ExclusiveGateway):
                self.failure_modes.append(FailureMode(
                    node_id=str(node.get_id()),
                    node_type="ExclusiveGateway",
                    description=f"Decision point failure at gateway {node.get_name() if node.get_name() else ''}",
                    historical_frequency=0.3
                ))

            elif isinstance(node, BPMN.ParallelGateway):
                self.failure_modes.append(FailureMode(
                    node_id=str(node.get_id()),
                    node_type="ParallelGateway",
                    description=f"Parallel execution failure at gateway {node.get_name() if node.get_name() else ''}",
                    historical_frequency=0.2
                ))

            elif isinstance(node, BPMN.Task):
                activity_name = node.get_name()
                stats = self.activity_stats.get(activity_name, {})
                frequency = stats.get('frequency', 0)
                completion_rate = stats.get('completion_rate', 100)
                historical_frequency = (100 - completion_rate) / 100 if frequency > 0 else 0.5

                self.failure_modes.append(FailureMode(
                    node_id=str(node.get_id()),
                    node_type="Task",
                    description=f"Activity failure in {activity_name}",
                    historical_frequency=historical_frequency
                ))

            elif isinstance(node, BPMN.SubProcess):
                self.failure_modes.append(FailureMode(
                    node_id=str(node.get_id()),
                    node_type="SubProcess",
                    description=f"Sub-process failure in {node.get_name() if node.get_name() else ''}",
                    historical_frequency=0.4
                ))

            # Add complexity-based failure modes for gateways with many connections
            if isinstance(node, (BPMN.ExclusiveGateway, BPMN.ParallelGateway)):
                if len(incoming) > 2 or len(outgoing) > 2:
                    self.failure_modes.append(FailureMode(
                        node_id=str(node.get_id()),
                        node_type="ComplexGateway",
                        description=f"High complexity risk at {node.get_name() if node.get_name() else ''} "
                                    f"({len(incoming)} inputs, {len(outgoing)} outputs)",
                        historical_frequency=0.6
                    ))


class EnhancedFMEA:
    def __init__(self, failure_modes: List[FailureMode], activity_stats: Dict):
        self.failure_modes = failure_modes
        self.activity_stats = activity_stats

    def calculate_severity(self, failure_mode: FailureMode) -> int:
        """Calculate severity based on node type and position in process"""
        base_severity = {
            'Task': 5,
            'SubProcess': 7,
            'ExclusiveGateway': 6,
            'ParallelGateway': 4,
            'ComplexGateway': 8
        }.get(failure_mode.node_type, 5)

        # Adjust based on downstream activities
        if failure_mode.downstream_activities:
            impact_factor = len(failure_mode.downstream_activities) / 5
            base_severity *= (1 + impact_factor)

        return min(int(base_severity), 10)

    def calculate_likelihood(self, failure_mode: FailureMode) -> int:
        """Calculate likelihood based on historical data and complexity"""
        base_likelihood = failure_mode.historical_frequency * 10

        # Adjust based on activity statistics
        if failure_mode.node_type in ['Task', 'SubProcess']:
            stats = self.activity_stats.get(failure_mode.node_id, {})
            variation_factor = len(stats.get('variations', set())) / 10
            base_likelihood *= (1 + variation_factor)

        return min(int(base_likelihood), 10)

    def calculate_detectability(self, failure_mode: FailureMode) -> int:
        """Calculate detectability based on node type and monitoring capability"""
        base_detectability = {
            'Task': 3,  # Easily detectable
            'SubProcess': 5,  # Moderately detectable
            'ExclusiveGateway': 7,  # Hard to detect decision errors
            'ParallelGateway': 4,
            'ComplexGateway': 8
        }.get(failure_mode.node_type, 5)

        # Adjust based on activity statistics
        if failure_mode.node_type in ['Task', 'SubProcess']:
            stats = self.activity_stats.get(failure_mode.node_id, {})
            if stats.get('frequency', 0) > 100:  # High frequency means better detection
                base_detectability -= 2

        return max(min(base_detectability, 10), 1)

    def assess_risk(self) -> List[Dict]:
        """Perform FMEA analysis for all failure modes"""
        results = []

        for failure_mode in self.failure_modes:
            severity = self.calculate_severity(failure_mode)
            likelihood = self.calculate_likelihood(failure_mode)
            detectability = self.calculate_detectability(failure_mode)
            rpn = severity * likelihood * detectability

            results.append({
                "failure_mode": failure_mode.description,
                "severity": severity,
                "likelihood": likelihood,
                "detectability": detectability,
                "rpn": rpn,
                "node_id": failure_mode.node_id,
                "recommendations": self._generate_recommendations(failure_mode, rpn)
            })

        return sorted(results, key=lambda x: x['rpn'], reverse=True)

    def _generate_recommendations(self, failure_mode: FailureMode, rpn: int) -> List[str]:
        """Generate recommendations based on failure mode and RPN"""
        recommendations = []

        if rpn > 200:
            recommendations.append("Immediate attention required - Consider process redesign")
        elif rpn > 100:
            recommendations.append("Implement additional controls and monitoring")

        if failure_mode.node_type == 'ExclusiveGateway':
            recommendations.append("Consider adding decision validation steps")
        elif failure_mode.node_type == 'ParallelGateway':
            recommendations.append("Implement synchronization checks")
        elif failure_mode.node_type == 'ComplexGateway':
            recommendations.append("Consider simplifying gateway logic")
        elif failure_mode.node_type in ['Task', 'SubProcess']:
            recommendations.append("Review standard operating procedures")

        return recommendations


def process_mining_with_risk_assessment(event_log, bpmn_graph):
    """Perform risk assessment on process model"""
    try:
        # Initialize analyzers
        risk_analyzer = ProcessRiskAnalyzer(event_log, bpmn_graph)
        risk_analyzer.analyze_bpmn_graph()

        # Perform FMEA
        fmea = EnhancedFMEA(risk_analyzer.failure_modes, risk_analyzer.activity_stats)
        risk_assessment_results = fmea.assess_risk()

        return bpmn_graph, risk_assessment_results

    except Exception as e:
        print(f"Error in risk assessment: {str(e)}")
        raise