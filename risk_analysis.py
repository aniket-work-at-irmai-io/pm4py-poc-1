# risk_analysis.py
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple, Any
import pm4py
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from pm4py.objects.bpmn.obj import BPMN
from pm4py.statistics.traces.generic.pandas import case_statistics

from bpmn_utils import BPMNGraphUtils


@dataclass
class FailureMode:
    """Data class for failure modes"""
    node_id: str
    node_type: str
    description: str
    upstream_activities: List[str]  # List of node IDs
    downstream_activities: List[str]  # List of node IDs
    critical_path: bool = False
    path_complexity: float = 0.0


class ProcessPathAnalyzer:
    """Analyzes BPMN paths and structural characteristics"""

    def __init__(self, bpmn_graph):
        self.bpmn_utils = BPMNGraphUtils(bpmn_graph)
        self.paths = self.bpmn_utils.get_all_paths()
        self.critical_path = self._identify_critical_path()  # Fixed method name to match private convention
        self.process_metrics = self.analyze_process_metrics()

    def _identify_critical_path(self) -> List[str]:
        """Identify critical path based on structural analysis"""
        paths_with_weights = []

        for path in self.paths:
            weight = self._calculate_path_weight(path)
            paths_with_weights.append((path, weight))

        if not paths_with_weights:
            return []

        return max(paths_with_weights, key=lambda x: x[1])[0]

    def analyze_process_metrics(self) -> Dict:
        """Analyze process-wide metrics"""
        metrics = {
            'total_nodes': len(self.bpmn_utils.get_all_nodes()),
            'gateways': len([n for n in self.bpmn_utils.get_all_nodes() if self.bpmn_utils.is_gateway(n.id)]),
            'tasks': len([n for n in self.bpmn_utils.get_all_nodes() if self.bpmn_utils.is_task(n.id)]),
            'events': len(self.bpmn_utils.get_start_nodes()) + len(self.bpmn_utils.get_end_nodes()),
            'parallel_splits': 0,
            'decision_points': 0,
            'path_lengths': [len(path) for path in self.paths],
            'max_parallel_paths': 0
        }

        # Count gateway types using node IDs instead of node objects
        for node in self.bpmn_utils.get_all_nodes():
            if hasattr(node, 'type') and 'parallel' in node.type.lower():
                metrics['parallel_splits'] += 1
                metrics['max_parallel_paths'] = max(metrics['max_parallel_paths'],
                                                    len(getattr(node, 'outgoing', [])))
            elif hasattr(node, 'type') and 'exclusive' in node.type.lower():
                metrics['decision_points'] += 1

        metrics['process_complexity'] = self.calculate_process_complexity(metrics)
        return metrics

    def calculate_process_complexity(self, metrics: Dict) -> float:
        """Calculate overall process complexity"""
        complexity_factors = {
            'decision_points': 0.3,
            'parallel_paths': 0.3,
            'path_variance': 0.2,
            'node_count': 0.2
        }

        # Normalize metrics
        max_nodes = 100  # baseline for normalization
        path_lengths = metrics['path_lengths']
        path_variance = np.var(path_lengths) if path_lengths else 0

        normalized_complexity = (
                (metrics['decision_points'] / max_nodes) * complexity_factors['decision_points'] +
                (metrics['max_parallel_paths'] / 10) * complexity_factors['parallel_paths'] +
                (path_variance / 100) * complexity_factors['path_variance'] +
                (metrics['total_nodes'] / max_nodes) * complexity_factors['node_count']
        )

        return normalized_complexity

    def _calculate_path_weight(self, path: List[str]) -> float:
        """Calculate path weight based on node types and connections using node IDs"""
        weight = 0
        for node_id in path:
            node = self.bpmn_utils.get_node_by_id(node_id)
            if not node:
                continue

            # Weight based on node type
            if self.bpmn_utils.is_task(node_id):
                weight += 1
            elif hasattr(node, 'type') and 'subprocess' in node.type.lower():
                weight += 2
            elif self.bpmn_utils.is_gateway(node_id):
                weight += len(getattr(node, 'outgoing', [])) * 0.5

            # Weight based on connections
            weight += len(getattr(node, 'incoming', [])) * 0.2
            weight += len(getattr(node, 'outgoing', [])) * 0.2

        return weight

    def _is_in_cycle(self, node_id: str) -> bool:
        """Check if node is part of a cycle using node IDs instead of objects"""
        visited = set()  # Set of node IDs

        def has_cycle(current_id: str, path: Set[str]) -> bool:
            if current_id in path:
                return True

            current_node = self.bpmn_utils.get_node_by_id(current_id)
            if not current_node:
                return False

            path.add(current_id)
            # Use node IDs from outgoing edges
            for target_id in current_node.outgoing:
                if has_cycle(target_id, path):
                    return True
            path.remove(current_id)
            return False

        return has_cycle(node_id, set())

    def calculate_path_complexity(self, node_id: str) -> float:
        """Calculate complexity score based on path characteristics using node IDs"""
        node = self.bpmn_utils.get_node_by_id(node_id)
        if not node:
            return 0.0

        # Get downstream and upstream nodes using IDs
        downstream_ids = self.bpmn_utils.get_downstream_nodes(node_id)
        upstream_ids = self.bpmn_utils.get_upstream_nodes(node_id)

        # Get actual nodes from IDs
        downstream_nodes = [self.bpmn_utils.get_node_by_id(n_id) for n_id in downstream_ids]
        upstream_nodes = [self.bpmn_utils.get_node_by_id(n_id) for n_id in upstream_ids]

        # Filter out None values
        downstream_nodes = [n for n in downstream_nodes if n is not None]
        upstream_nodes = [n for n in upstream_nodes if n is not None]

        complexity_factors = {
            'downstream_decisions': sum(1 for n in downstream_nodes
                                        if hasattr(n, 'type') and 'exclusive' in n.type.lower()),
            'downstream_parallels': sum(1 for n in downstream_nodes
                                        if hasattr(n, 'type') and 'parallel' in n.type.lower()),
            'upstream_merges': sum(1 for n in upstream_nodes
                                   if self.bpmn_utils.is_gateway(n.id)),
            'path_count': sum(1 for path in self.paths if node_id in path),
            'is_in_cycle': self._is_in_cycle(node_id)
        }

        # Calculate weighted complexity
        complexity = (
                complexity_factors['downstream_decisions'] * 0.3 +
                complexity_factors['downstream_parallels'] * 0.2 +
                complexity_factors['upstream_merges'] * 0.2 +
                complexity_factors['path_count'] * 0.2 +
                (1.0 if complexity_factors['is_in_cycle'] else 0.0) * 0.1
        )

        return complexity

    def _get_downstream_nodes(self, node_id: str) -> Set[str]:
        """Get all nodes downstream from given node"""
        return {n.id for n in self.bpmn_utils.get_downstream_nodes(node_id)}

    def _get_upstream_nodes(self, node_id: str) -> Set[str]:
        """Get all nodes upstream from given node"""
        return {n.id for n in self.bpmn_utils.get_upstream_nodes(node_id)}


class EventLogAnalyzer:
    """Analyzes event log for historical patterns and statistics"""

    def __init__(self, event_log):
        self.event_log = event_log
        self.activity_statistics = self._compute_activity_statistics()
        self.case_statistics = self._compute_case_statistics()

    def _compute_case_statistics(self) -> Dict:
        """Compute statistics per case"""
        stats = {
            'case_durations': [],
            'case_activities': [],
            'case_resources': [],
            'throughput_times': [],
            'case_variants': defaultdict(int)
        }

        for trace in self.event_log:
            # Skip empty traces
            if not trace:
                continue

            # Calculate case duration
            start_time = min(event['time:timestamp'] for event in trace if 'time:timestamp' in event)
            end_time = max(event['time:timestamp'] for event in trace if 'time:timestamp' in event)
            duration = (end_time - start_time).total_seconds()
            stats['case_durations'].append(duration)

            # Count activities and resources per case
            activities = set()
            resources = set()
            activity_sequence = []

            for event in trace:
                if 'concept:name' in event:
                    activities.add(event['concept:name'])
                    activity_sequence.append(event['concept:name'])
                if 'org:resource' in event:
                    resources.add(event['org:resource'])

            stats['case_activities'].append(len(activities))
            stats['case_resources'].append(len(resources))
            stats['throughput_times'].append(duration)

            # Track case variants (unique sequences of activities)
            variant_key = ','.join(activity_sequence)
            stats['case_variants'][variant_key] += 1

        # Calculate aggregate statistics
        if stats['case_durations']:
            stats['avg_case_duration'] = np.mean(stats['case_durations'])
            stats['median_case_duration'] = np.median(stats['case_durations'])
            stats['min_case_duration'] = min(stats['case_durations'])
            stats['max_case_duration'] = max(stats['case_durations'])

        if stats['case_activities']:
            stats['avg_activities_per_case'] = np.mean(stats['case_activities'])
            stats['max_activities_per_case'] = max(stats['case_activities'])

        if stats['case_resources']:
            stats['avg_resources_per_case'] = np.mean(stats['case_resources'])
            stats['max_resources_per_case'] = max(stats['case_resources'])

        # Calculate variant statistics
        total_cases = sum(stats['case_variants'].values())
        stats['variant_statistics'] = {
            'total_variants': len(stats['case_variants']),
            'most_common_variants': sorted(
                [(variant, count) for variant, count in stats['case_variants'].items()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'variant_coverage': {
                variant: (count / total_cases * 100)
                for variant, count in stats['case_variants'].items()
            }
        }

        return stats

    def calculate_failure_likelihood(self, node_id: str, path_complexity: float) -> float:
        """Calculate likelihood of failure based on historical data and path complexity"""
        # Convert node to its ID if it's a BPMNNode object
        node_id = node_id.id if hasattr(node_id, 'id') else str(node_id)

        stats = self.activity_statistics.get(node_id, {})

        # Historical factors
        rework_rate = stats.get('rework_rate', 0)
        duration_variance = stats.get('duration_variance', 0)
        resource_diversity = stats.get('resource_diversity', 1)

        # Normalize factors
        normalized_variance = min(duration_variance / (3600 * 24), 1)  # Normalize to day
        normalized_diversity = min(resource_diversity / 10, 1)

        # Calculate likelihood components
        historical_factor = (rework_rate * 0.4 + normalized_variance * 0.3 + normalized_diversity * 0.3)
        complexity_factor = min(path_complexity, 1.0)

        # Combined likelihood
        likelihood = (historical_factor * 0.7 + complexity_factor * 0.3)

        return likelihood

    def calculate_detection_capability(self, node_id: str) -> float:
        """Calculate how easily failures can be detected"""
        # Convert node to its ID if it's a BPMNNode object
        node_id = node_id.id if hasattr(node_id, 'id') else str(node_id)

        stats = self.activity_statistics.get(node_id, {})

        # Rest of the method remains the same
        frequency = stats.get('frequency', 0)
        automated = stats.get('automated', False)
        resource_diversity = stats.get('resource_diversity', 1)
        monitoring_ratio = stats.get('monitoring_ratio', 0)
        control_ratio = stats.get('control_ratio', 0)

        automation_factor = 0.8 if automated else 0.4
        frequency_factor = min(frequency / 1000, 1.0)
        resource_factor = min(resource_diversity / 5, 1.0)
        monitoring_factor = monitoring_ratio * 0.8 + control_ratio * 0.2

        if stats.get('durations', []):
            duration_std = stats.get('duration_stddev', 0)
            mean_duration = stats.get('avg_duration', 1)
            coefficient_variation = duration_std / mean_duration if mean_duration > 0 else 1
            consistency_factor = 1 - min(coefficient_variation, 1)
        else:
            consistency_factor = 0.5

        detectability = (
                automation_factor * 0.25 +
                frequency_factor * 0.15 +
                resource_factor * 0.15 +
                monitoring_factor * 0.25 +
                consistency_factor * 0.20
        )

        return detectability

    def _compute_activity_statistics(self) -> Dict:
        """Compute comprehensive activity statistics using node IDs"""
        stats = defaultdict(lambda: {
            'frequency': 0,
            'avg_duration': timedelta(0),
            'durations': [],
            'resources': defaultdict(int),
            'next_activities': defaultdict(int),
            'automated': False,
            'rework_frequency': 0,
            'failure_patterns': defaultdict(int),
            'control_points': 0,
            'monitoring_events': 0
        })

        for trace in self.event_log:
            prev_event = None
            activities_in_case = set()  # Use activity IDs instead of BPMNNode objects

            for event in trace:
                # Get activity ID
                activity_id = event['concept:name']
                stats[activity_id]['frequency'] += 1

                # Track duration
                if 'time:timestamp' in event:
                    if prev_event and 'time:timestamp' in prev_event:
                        duration = event['time:timestamp'] - prev_event['time:timestamp']
                        stats[activity_id]['durations'].append(duration.total_seconds())

                # Track resources
                if 'org:resource' in event:
                    stats[activity_id]['resources'][event['org:resource']] += 1

                # Track control points
                if any(attr.startswith('check_') or attr.startswith('verify_') for attr in event.keys()):
                    stats[activity_id]['control_points'] += 1

                # Track monitoring events
                if any(attr.startswith('monitor_') or attr.startswith('measure_') for attr in event.keys()):
                    stats[activity_id]['monitoring_events'] += 1

                # Track activity transitions using IDs
                if prev_event:
                    prev_activity_id = prev_event['concept:name']
                    stats[activity_id]['next_activities'][prev_activity_id] += 1

                # Check for rework using activity IDs
                if activity_id in activities_in_case:
                    stats[activity_id]['rework_frequency'] += 1
                activities_in_case.add(activity_id)

                prev_event = event

        # Calculate aggregate statistics
        for activity_id, data in stats.items():
            if data['durations']:
                data['avg_duration'] = np.mean(data['durations'])
                data['duration_stddev'] = np.std(data['durations'])
                data['duration_variance'] = np.var(data['durations'])

            total_executions = sum(data['resources'].values())
            if total_executions > 0:
                data['resource_distribution'] = {
                    resource: count / total_executions
                    for resource, count in data['resources'].items()
                }
                data['resource_diversity'] = len(data['resources'])

            total_transitions = sum(data['next_activities'].values())
            if total_transitions > 0:
                data['transition_probabilities'] = {
                    act: count / total_transitions
                    for act, count in data['next_activities'].items()
                }

            data['automated'] = len(data['resources']) <= 1
            data['monitoring_ratio'] = data['monitoring_events'] / data['frequency'] if data['frequency'] > 0 else 0
            data['control_ratio'] = data['control_points'] / data['frequency'] if data['frequency'] > 0 else 0
            data['rework_rate'] = data['rework_frequency'] / data['frequency'] if data['frequency'] > 0 else 0

        return dict(stats)

    def calculate_failure_likelihood(self, node_id: str, path_complexity: float) -> float:
        """Calculate likelihood of failure based on historical data and path complexity"""
        stats = self.activity_statistics.get(node_id, {})

        # Historical factors
        rework_rate = stats.get('rework_rate', 0)
        duration_variance = stats.get('duration_variance', 0)
        resource_diversity = stats.get('resource_diversity', 1)

        # Normalize factors
        normalized_variance = min(duration_variance / (3600 * 24), 1)  # Normalize to day
        normalized_diversity = min(resource_diversity / 10, 1)

        # Calculate likelihood components
        historical_factor = (rework_rate * 0.4 + normalized_variance * 0.3 +
                             normalized_diversity * 0.3)
        complexity_factor = min(path_complexity, 1.0)

        # Combined likelihood
        likelihood = (historical_factor * 0.7 + complexity_factor * 0.3)

        return likelihood

    def calculate_detection_capability(self, node_id: str) -> float:
        """Calculate how easily failures can be detected based on process characteristics"""
        stats = self.activity_statistics.get(node_id, {})

        # Base detection factors
        frequency = stats.get('frequency', 0)
        automated = stats.get('automated', False)
        resource_diversity = stats.get('resource_diversity', 1)
        monitoring_ratio = stats.get('monitoring_ratio', 0)
        control_ratio = stats.get('control_ratio', 0)

        # Calculate detection components
        automation_factor = 0.8 if automated else 0.4  # Automated activities are easier to monitor

        frequency_factor = min(frequency / 1000, 1.0)  # Normalize to reasonable max

        resource_factor = min(resource_diversity / 5, 1.0)  # More resources = more detection points

        monitoring_factor = monitoring_ratio * 0.8 + control_ratio * 0.2

        # Duration consistency factor
        if stats.get('durations', []):
            duration_std = stats.get('duration_stddev', 0)
            mean_duration = stats.get('avg_duration', 1)
            coefficient_variation = duration_std / mean_duration if mean_duration > 0 else 1
            consistency_factor = 1 - min(coefficient_variation, 1)  # Lower variation = easier detection
        else:
            consistency_factor = 0.5

        # Combine factors with weights
        detectability = (
                automation_factor * 0.25 +
                frequency_factor * 0.15 +
                resource_factor * 0.15 +
                monitoring_factor * 0.25 +
                consistency_factor * 0.20
        )

        return detectability  # Returns value between 0-1, where 1 means highly detectable


import logging
from typing import List, Dict, Any
import traceback

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class EnhancedRiskAnalyzer:
    """Main class for enhanced risk analysis"""

    def __init__(self, event_log, bpmn_graph):
        logger.info("Initializing EnhancedRiskAnalyzer")
        try:
            self.event_log = event_log
            self.bpmn_graph = bpmn_graph
            logger.debug(f"BPMN Graph type: {type(bpmn_graph)}")

            logger.info("Initializing BPMNGraphUtils")
            self.bpmn_utils = BPMNGraphUtils(bpmn_graph)

            logger.info("Initializing ProcessPathAnalyzer")
            self.path_analyzer = ProcessPathAnalyzer(bpmn_graph)

            logger.info("Initializing EventLogAnalyzer")
            self.log_analyzer = EventLogAnalyzer(event_log)

            logger.info("Identifying failure modes")
            self.failure_modes = self._identify_failure_modes()
            logger.debug(f"Found {len(self.failure_modes)} failure modes")

            logger.info("Getting process metrics")
            self.process_metrics = self.path_analyzer.process_metrics
            logger.debug(f"Process metrics: {self.process_metrics}")

        except Exception as e:
            logger.error(f"Error in initialization: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _identify_failure_modes(self) -> List[FailureMode]:
        """Identify potential failure modes from BPMN structure"""
        failure_modes = []
        logger.info("Starting failure mode identification")

        for node in self.bpmn_utils.get_all_nodes():
            logger.debug(f"Processing node: {node.id}")

            if self.bpmn_utils.is_task(node.id):
                logger.debug(f"Node {node.id} is a task, analyzing...")

                try:
                    # Get upstream and downstream activities
                    upstream_ids = list(self.bpmn_utils.get_upstream_nodes(node.id))
                    downstream_ids = list(self.bpmn_utils.get_downstream_nodes(node.id))

                    # Check if node is in critical path
                    is_critical = node.id in self.path_analyzer.critical_path
                    logger.debug(f"Node {node.id} critical path status: {is_critical}")

                    # Calculate path complexity
                    path_complexity = self.path_analyzer.calculate_path_complexity(node.id)
                    logger.debug(f"Node {node.id} path complexity: {path_complexity}")

                    # Create failure mode
                    failure_mode = FailureMode(
                        node_id=node.id,
                        node_type=node.type,
                        description=f"Failure in {node.name or node.id}",
                        upstream_activities=upstream_ids,
                        downstream_activities=downstream_ids,
                        critical_path=is_critical,
                        path_complexity=path_complexity
                    )

                    failure_modes.append(failure_mode)
                    logger.debug(f"Added failure mode for node {node.id}")

                except Exception as e:
                    logger.error(f"Error processing node {node.id}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

        logger.info(f"Identified {len(failure_modes)} failure modes")
        return failure_modes

    def analyze_risks(self) -> List[Dict]:
        """Perform comprehensive risk analysis"""
        risk_assessment = []
        logger.info("Starting risk analysis")

        try:
            for failure_mode in self.failure_modes:
                logger.debug(f"Analyzing failure mode: {failure_mode.description}")

                try:
                    # Get node and verify it exists
                    node = self.bpmn_utils.get_node_by_id(failure_mode.node_id)
                    if not node:
                        logger.warning(f"Node not found for failure mode: {failure_mode.node_id}")
                        continue

                    # Calculate all risk components with proper error handling
                    try:
                        logger.debug(f"Calculating severity for {failure_mode.node_id}")
                        severity = self.calculate_severity(failure_mode)
                    except Exception as e:
                        logger.error(f"Error calculating severity: {str(e)}")
                        continue

                    try:
                        logger.debug(f"Calculating likelihood for {failure_mode.node_id}")
                        likelihood = self.calculate_likelihood(failure_mode)
                    except Exception as e:
                        logger.error(f"Error calculating likelihood: {str(e)}")
                        continue

                    try:
                        logger.debug(f"Calculating detectability for {failure_mode.node_id}")
                        detectability = self.calculate_detectability(failure_mode)
                    except Exception as e:
                        logger.error(f"Error calculating detectability: {str(e)}")
                        continue

                    # Only proceed if all calculations were successful
                    if all(isinstance(x, (int, float)) for x in [severity, likelihood, detectability]):
                        rpn = (severity * likelihood * detectability) / 100
                        logger.debug(f"Calculated RPN for {failure_mode.node_id}: {rpn}")

                        risk_assessment.append({
                            'failure_mode': failure_mode.description,
                            'node_id': failure_mode.node_id,
                            'node_type': failure_mode.node_type,
                            'severity': round(severity, 2),
                            'likelihood': round(likelihood, 2),
                            'detectability': round(detectability, 2),
                            'rpn': round(rpn, 2),
                            'critical_path': failure_mode.critical_path,
                            'path_complexity': failure_mode.path_complexity,
                            'structural_details': {
                                'downstream_activities': len(failure_mode.downstream_activities),
                                'upstream_activities': len(failure_mode.upstream_activities),
                                'process_position': 'Critical Path' if failure_mode.critical_path else 'Normal Path'
                            }
                        })
                        logger.debug(f"Added risk assessment for {failure_mode.node_id}")

                except Exception as e:
                    logger.error(f"Error analyzing failure mode {failure_mode.node_id}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

            logger.info(f"Completed risk analysis for {len(risk_assessment)} nodes")
            return sorted(risk_assessment, key=lambda x: x['rpn'], reverse=True)

        except Exception as e:
            logger.error(f"Error in risk analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def calculate_severity(self, failure_mode: FailureMode) -> float:
        """Calculate severity based on structural characteristics"""
        node = self.bpmn_utils.get_node_by_id(failure_mode.node_id)
        if not node:
            return 0.0

        # Calculate severity components with proper node ID handling
        severity_factors = {
            'gateway_impact': self._calculate_gateway_impact(failure_mode.node_id),
            'path_criticality': self._calculate_path_criticality(failure_mode),
            'data_dependencies': self._calculate_data_dependencies(failure_mode.node_id),
            'message_impact': self._calculate_message_flow_impact(failure_mode.node_id)
        }

        # Calculate weighted severity
        severity = (
                severity_factors['gateway_impact'] * 0.3 +
                severity_factors['path_criticality'] * 0.3 +
                severity_factors['data_dependencies'] * 0.2 +
                severity_factors['message_impact'] * 0.2
        )

        return min(severity * 10, 10)

    def calculate_likelihood(self, failure_mode: FailureMode) -> float:
        """Calculate likelihood of failure occurrence"""
        node = self.bpmn_utils.get_node_by_id(failure_mode.node_id)
        if not node:
            return 0.0

        pattern_likelihood = self._calculate_pattern_likelihood(failure_mode.node_id)
        historical_likelihood = self.log_analyzer.calculate_failure_likelihood(
            failure_mode.node_id,
            failure_mode.path_complexity
        )

        likelihood = (pattern_likelihood * 0.6 + historical_likelihood * 0.4)
        return min(likelihood * 10, 10)

    def calculate_detectability(self, failure_mode: FailureMode) -> float:
        """Calculate how easily failures can be detected"""
        try:
            node = self.bpmn_utils.get_node_by_id(failure_mode.node_id)
            if not node:
                logger.warning(f"Node not found for detectability calculation: {failure_mode.node_id}")
                return 0.0

            # Calculate all detectability factors with proper error handling
            detectability_factors = {}

            try:
                detectability_factors['monitoring_points'] = self._count_monitoring_points(failure_mode.node_id)
            except Exception as e:
                logger.error(f"Error calculating monitoring points: {str(e)}")
                detectability_factors['monitoring_points'] = 0.0

            try:
                detectability_factors['event_monitoring'] = self._analyze_event_monitoring(failure_mode.node_id)
            except Exception as e:
                logger.error(f"Error analyzing event monitoring: {str(e)}")
                detectability_factors['event_monitoring'] = 0.0

            try:
                detectability_factors['gateway_detection'] = self._analyze_gateway_detection(failure_mode.node_id)
            except Exception as e:
                logger.error(f"Error analyzing gateway detection: {str(e)}")
                detectability_factors['gateway_detection'] = 0.0

            try:
                detectability_factors['message_monitoring'] = self._analyze_message_monitoring(failure_mode.node_id)
            except Exception as e:
                logger.error(f"Error analyzing message monitoring: {str(e)}")
                detectability_factors['message_monitoring'] = 0.0

            # Calculate overall detectability if we have any valid factors
            if detectability_factors:
                detectability = sum(detectability_factors.values()) / len(detectability_factors)
                # Convert to detectability score (inverse of detection ease)
                detectability_score = min((1 - detectability) * 10, 10)
                logger.debug(f"Calculated detectability score for {failure_mode.node_id}: {detectability_score}")
                return detectability_score
            else:
                logger.warning(f"No valid detectability factors for {failure_mode.node_id}")
                return 0.0

        except Exception as e:
            logger.error(f"Error in detectability calculation for {failure_mode.node_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

    def _calculate_gateway_impact(self, node_id: str) -> float:
        """Calculate impact based on gateway patterns"""
        # Get downstream nodes as IDs
        downstream_ids = self.bpmn_utils.get_downstream_nodes(node_id)
        impact = 0.0

        for downstream_id in downstream_ids:
            downstream_node = self.bpmn_utils.get_node_by_id(downstream_id)
            if downstream_node and hasattr(downstream_node, 'type'):
                node_type = downstream_node.type.lower()
                if 'exclusive' in node_type:
                    impact += 0.5
                elif 'parallel' in node_type:
                    impact += 0.3
                elif 'inclusive' in node_type:
                    impact += 0.7

        return min(impact, 1.0)

    def _calculate_path_criticality(self, failure_mode: FailureMode) -> float:
        """Calculate criticality based on path characteristics"""
        if failure_mode.critical_path:
            alternate_paths = [p for p in self.path_analyzer.paths
                               if failure_mode.node_id not in p]
            if not alternate_paths:
                return 1.0
            return 0.7
        return 0.3

    def _calculate_data_dependencies(self, node_id: str) -> float:
        """Calculate impact based on data dependencies"""
        node = self.bpmn_utils.get_node_by_id(node_id)
        if not node:
            return 0.0

        # Count connections only if we have a valid node
        total_connections = len(getattr(node, 'incoming', [])) + len(getattr(node, 'outgoing', []))
        return min(total_connections / 10, 1.0)

    def _calculate_message_flow_impact(self, node_id: str) -> float:
        """Calculate impact based on message flows"""
        node = self.bpmn_utils.get_node_by_id(node_id)
        if not node:
            return 0.0

        # Count outgoing connections only if we have a valid node
        outgoing_count = len(getattr(node, 'outgoing', []))
        return min(outgoing_count / 5, 1.0)

    def _calculate_pattern_likelihood(self, node_id: str) -> float:
        """Calculate likelihood based on BPMN patterns"""
        likelihood_factors = {
            'loops': self._identify_loop_patterns(node_id),
            'gateway_complexity': self._calculate_gateway_complexity(node_id),
            'event_handlers': self._count_event_handlers(node_id),
            'boundary_events': self._analyze_boundary_events(node_id)
        }

        return sum(likelihood_factors.values()) / len(likelihood_factors)

    def _identify_loop_patterns(self, node_id: str) -> float:
        """Identify loop patterns in the process"""
        return 1.0 if self.path_analyzer._is_in_cycle(node_id) else 0.0

    def _calculate_gateway_complexity(self, node_id: str) -> float:
        """Calculate complexity of surrounding gateways"""
        downstream_ids = self.bpmn_utils.get_downstream_nodes(node_id)
        gateway_count = 0

        for downstream_id in downstream_ids:
            downstream_node = self.bpmn_utils.get_node_by_id(downstream_id)
            if downstream_node and self.bpmn_utils.is_gateway(downstream_id):
                gateway_count += 1

        return min(gateway_count / 5, 1.0)

    def _count_event_handlers(self, node_id: str) -> float:
        """Count event handlers connected to the node"""
        node = self.bpmn_utils.get_node_by_id(node_id)
        if not node:
            return 0.0

        event_count = 0
        for incoming_id in node.incoming:
            source_node = self.bpmn_utils.get_node_by_id(incoming_id)
            if source_node and hasattr(source_node, 'type') and 'event' in source_node.type.lower():
                event_count += 1

        return min(event_count / 3, 1.0)

    def _analyze_boundary_events(self, node_id: str) -> float:
        """Analyze boundary events attached to the node"""
        node = self.bpmn_utils.get_node_by_id(node_id)
        if not node:
            return 0.0

        # For incoming connections over 2, consider it may have boundary events
        return 0.5 if len(getattr(node, 'incoming', [])) > 2 else 0.0

    def _count_monitoring_points(self, node_id: str) -> float:
        """Count monitoring points in the process"""
        downstream_ids = self.bpmn_utils.get_downstream_nodes(node_id)
        monitoring_points = 0

        for downstream_id in downstream_ids:
            downstream_node = self.bpmn_utils.get_node_by_id(downstream_id)
            if downstream_node and hasattr(downstream_node, 'type') and 'gateway' in downstream_node.type.lower():
                monitoring_points += 1

        return min(monitoring_points / 3, 1.0)

    def _analyze_event_monitoring(self, node_id: str) -> float:
        """Analyze event-based monitoring capabilities"""
        node = self.bpmn_utils.get_node_by_id(node_id)
        if not node:
            return 0.0

        event_count = 0
        for outgoing_id in node.outgoing:
            target_node = self.bpmn_utils.get_node_by_id(outgoing_id)
            if target_node and hasattr(target_node, 'type') and 'event' in target_node.type.lower():
                event_count += 1

        return min(event_count / 2, 1.0)

    def _analyze_gateway_detection(self, node_id: str) -> float:
        """Analyze gateway-based detection capabilities"""
        downstream_ids = self.bpmn_utils.get_downstream_nodes(node_id)
        exclusive_gateways = 0

        for downstream_id in downstream_ids:
            downstream_node = self.bpmn_utils.get_node_by_id(downstream_id)
            if downstream_node and hasattr(downstream_node, 'type') and 'exclusive' in downstream_node.type.lower():
                exclusive_gateways += 1

        return min(exclusive_gateways / 2, 1.0)

    def _analyze_message_monitoring(self, node_id: str) -> float:
        """Analyze message-based monitoring capabilities"""
        node = self.bpmn_utils.get_node_by_id(node_id)
        if not node:
            return 0.0

        outgoing_count = len(getattr(node, 'outgoing', []))
        return min(outgoing_count / 4, 1.0)

    def _generate_recommendations(self, severity: float, likelihood: float,
                                  detectability: float, failure_mode: FailureMode) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []

        node = self.bpmn_utils.get_node_by_id(failure_mode.node_id)
        if not node:
            return recommendations

        # High severity recommendations
        if severity > 7:
            recommendations.append(f"Critical activity '{failure_mode.description}' requires immediate attention")
            recommendations.append("Consider adding redundancy or fallback mechanisms")

        # High likelihood recommendations
        if likelihood > 7:
            recommendations.append("Implement additional validation checks")
            recommendations.append("Consider process simplification to reduce failure points")

        # Poor detectability recommendations
        if detectability > 7:
            recommendations.append("Add monitoring points for better failure detection")
            recommendations.append("Implement automated testing and verification")

        # Critical path recommendations
        if failure_mode.critical_path:
            recommendations.append("Consider parallel processing to reduce critical path risk")

        # Complex path recommendations
        if failure_mode.path_complexity > 0.7:
            recommendations.append("Simplify process flow to reduce complexity")
            recommendations.append("Add intermediate validation steps")

        return recommendations


def process_mining_with_risk_assessment(event_log, bpmn_graph):
    """Main function to perform risk assessment on process model"""
    try:
        risk_analyzer = EnhancedRiskAnalyzer(event_log, bpmn_graph)
        risk_assessment_results = risk_analyzer.analyze_risks()
        return bpmn_graph, risk_assessment_results
    except Exception as e:
        print(f"Error in risk assessment: {str(e)}")
        raise