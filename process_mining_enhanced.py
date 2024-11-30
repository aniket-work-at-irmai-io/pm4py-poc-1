# process_mining_enhanced.py
import pandas as pd
import pm4py
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pm4py.algo.filtering.log import timestamp as timestamp_filter
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from risk_analysis import ProcessRiskAnalyzer, EnhancedFMEA

class FXProcessMining:
    def __init__(self, event_log_path: str, separator: str = ';'):
        self.raw_df = pd.read_csv(event_log_path, sep=separator)
        self.event_log = None
        self.process_tree = None
        self.process_model = None  # This will hold the Petri net
        self.initial_marking = None
        self.final_marking = None
        self.activity_stats = None

    def preprocess_data(self) -> None:
        """Step 2: Data Preprocessing"""
        # Remove incomplete cases (missing mandatory fields)
        self.raw_df = self.raw_df.dropna(subset=['case_id', 'activity', 'timestamp'])

        # Standardize activity names
        self.raw_df['activity'] = self.raw_df['activity'].str.strip().str.title()

        # Convert timestamp to datetime
        self.raw_df['timestamp'] = pd.to_datetime(self.raw_df['timestamp'])

        # Sort by case_id and timestamp
        self.raw_df = self.raw_df.sort_values(['case_id', 'timestamp'])

        # Convert to PM4Py event log
        self.event_log = pm4py.format_dataframe(
            self.raw_df,
            case_id='case_id',
            activity_key='activity',
            timestamp_key='timestamp'
        )
        self.event_log = pm4py.convert_to_event_log(self.event_log)

    def discover_process(self) -> None:
        """Step 3: Process Discovery"""
        # First get the process tree
        self.process_tree = inductive_miner.apply(self.event_log)

        # Then convert to Petri net for analysis
        self.process_model, self.initial_marking, self.final_marking = \
            pm4py.convert_to_petri_net(self.process_tree)

    def check_conformance(self) -> Dict:
        """Step 4: Conformance Checking"""
        if not self.process_model:
            raise ValueError("Process model not discovered. Run discover_process() first.")

        # Token-based replay
        replayed_traces = token_replay.apply(
            self.event_log,
            self.process_model,
            self.initial_marking,
            self.final_marking
        )

        # Calculate conformance metrics
        conformance_metrics = {
            'fitness': np.mean([trace['trace_fitness'] for trace in replayed_traces]),
            'completed_traces': sum(1 for trace in replayed_traces if trace['trace_is_fit']),
            'total_traces': len(replayed_traces)
        }

        return conformance_metrics

    def _get_cases_with_sequence(self, act1: str, act2: str) -> List[Dict]:
        """Helper method to get cases containing a specific sequence of activities"""
        cases = []
        for trace in self.event_log:
            case_id = trace.attributes['concept:name']
            activities = [event['concept:name'] for event in trace]

            # Find all occurrences of act1 in the trace
            for i in range(len(activities)-1):
                if activities[i] == act1 and activities[i+1] == act2:
                    # Get timestamps
                    act1_time = trace[i]['time:timestamp']
                    act2_time = trace[i+1]['time:timestamp']
                    duration = act2_time - act1_time

                    cases.append({
                        'case_id': case_id,
                        'duration': duration
                    })
                    break  # Only consider first occurrence in each case

        return cases

    def analyze_risks(self) -> Dict:
        """Analyze process risks using ProcessRiskAnalyzer and EnhancedFMEA"""
        if not self.process_tree:
            raise ValueError("Process model not discovered. Run discover_process() first.")

        # Convert process tree to BPMN for risk analysis
        bpmn_graph = pm4py.convert_to_bpmn(self.process_tree)

        # Initialize ProcessRiskAnalyzer
        risk_analyzer = ProcessRiskAnalyzer(
            event_log=self.event_log,
            bpmn_graph=bpmn_graph
        )

        # Analyze BPMN graph for failure modes
        risk_analyzer.analyze_bpmn_graph()

        # Get activity statistics
        self.activity_stats = risk_analyzer.activity_stats

        # Initialize EnhancedFMEA with failure modes and statistics
        fmea = EnhancedFMEA(
            failure_modes=risk_analyzer.failure_modes,
            activity_stats=self.activity_stats
        )

        # Perform FMEA analysis
        risk_assessment = fmea.assess_risk()

        # Calculate process metrics
        process_metrics = {
            'total_activities': len(self.activity_stats),
            'high_risk_activities': len([r for r in risk_assessment if r['rpn'] > 200]),
            'medium_risk_activities': len([r for r in risk_assessment if 100 < r['rpn'] <= 200]),
            'low_risk_activities': len([r for r in risk_assessment if r['rpn'] <= 100])
        }

        return {
            'risk_assessment': risk_assessment,
            'process_metrics': process_metrics
        }

    def generate_report(self) -> Dict:
        """Generate comprehensive process mining report including risks"""
        # Get risk analysis results
        risk_results = self.analyze_risks()

        return {
            'conformance': self.check_conformance(),
            'performance': self.analyze_performance(),
            'root_causes': self.perform_root_cause_analysis(),
            'improvements': self.suggest_improvements(),
            'risks': risk_results
        }


    def analyze_performance(self) -> Dict:
        """Step 5: Performance Analysis"""
        # Calculate case durations
        case_durations = case_statistics.get_all_case_durations(self.event_log)

        # Calculate throughput time statistics
        performance_metrics = {
            'avg_case_duration': np.mean(case_durations),
            'median_case_duration': np.median(case_durations),
            'min_case_duration': np.min(case_durations),
            'max_case_duration': np.max(case_durations)
        }

        # Identify bottlenecks
        from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
        from pm4py.statistics.start_activities.log import get as start_activities
        from pm4py.statistics.end_activities.log import get as end_activities

        dfg = dfg_discovery.apply(self.event_log)
        start_acts = start_activities.get_start_activities(self.event_log)
        end_acts = end_activities.get_end_activities(self.event_log)

        # Find activities with longest waiting times
        performance_metrics['bottlenecks'] = self._identify_bottlenecks(dfg)

        return performance_metrics


    def _identify_bottlenecks(self, dfg) -> List[Dict]:
        """Helper method to identify process bottlenecks"""
        bottlenecks = []
        avg_time_between_activities = {}

        for (act1, act2), freq in dfg.items():
            # Calculate average time between activities
            relevant_cases = self._get_cases_with_sequence(act1, act2)
            if relevant_cases:
                avg_time = np.mean([case['duration'].total_seconds()
                                    for case in relevant_cases])
                avg_time_between_activities[(act1, act2)] = avg_time

                if avg_time > np.median(list(avg_time_between_activities.values())):
                    bottlenecks.append({
                        'source': act1,
                        'target': act2,
                        'avg_duration': avg_time,
                        'frequency': freq
                    })

        return sorted(bottlenecks, key=lambda x: x['avg_duration'], reverse=True)

    def perform_root_cause_analysis(self) -> Dict:
        """Step 6: Root Cause Analysis"""
        analysis_results = {
            'variant_analysis': self._analyze_variants(),
            'attribute_impact': self._analyze_attribute_impact(),
            'temporal_patterns': self._analyze_temporal_patterns()
        }
        return analysis_results

    def _analyze_variants(self) -> List[Dict]:
        """Analyze process variants"""
        from pm4py.statistics.traces.generic.log import case_statistics

        variants = case_statistics.get_variant_statistics(self.event_log)
        return sorted(variants, key=lambda x: x['count'], reverse=True)

    def _analyze_attribute_impact(self) -> Dict:
        """Analyze impact of case attributes on performance"""
        impacts = {}
        for attribute in ['resource', 'currency_pair']:
            if attribute in self.raw_df.columns:
                attribute_stats = self.raw_df.groupby(attribute).agg({
                    'case_id': 'count',
                    'timestamp': lambda x: (x.max() - x.min()).total_seconds()
                }).reset_index()
                impacts[attribute] = attribute_stats.to_dict('records')
        return impacts

    def _analyze_temporal_patterns(self) -> Dict:
        """Analyze temporal patterns in the process"""
        self.raw_df['hour'] = self.raw_df['timestamp'].dt.hour
        self.raw_df['day_of_week'] = self.raw_df['timestamp'].dt.day_name()

        temporal_patterns = {
            'hourly_distribution': self.raw_df.groupby('hour')['case_id'].count().to_dict(),
            'daily_distribution': self.raw_df.groupby('day_of_week')['case_id'].count().to_dict()
        }
        return temporal_patterns

    def suggest_improvements(self) -> List[Dict]:
        """Step 7: Process Enhancement"""
        # Get performance metrics
        performance = self.analyze_performance()
        conformance = self.check_conformance()
        root_causes = self.perform_root_cause_analysis()

        improvements = []

        # Analyze bottlenecks
        for bottleneck in performance['bottlenecks'][:3]:
            improvements.append({
                'type': 'bottleneck',
                'location': f"{bottleneck['source']} → {bottleneck['target']}",
                'issue': f"High average duration: {bottleneck['avg_duration']:.2f} seconds",
                'recommendation': "Consider adding resources or automation"
            })

        # Analyze conformance issues
        if conformance['fitness'] < 0.95:
            improvements.append({
                'type': 'conformance',
                'issue': f"Low process fitness: {conformance['fitness']:.2f}",
                'recommendation': "Review and update process documentation and training"
            })

        # Analyze variant patterns
        variants = root_causes['variant_analysis']
        if len(variants) > 10:  # High variability
            improvements.append({
                'type': 'standardization',
                'issue': f"High process variability: {len(variants)} variants",
                'recommendation': "Standardize common process paths and investigate deviations"
            })

        return improvements


    def generate_report(self) -> Dict:
        """Generate comprehensive process mining report"""
        return {
            'conformance': self.check_conformance(),
            'performance': self.analyze_performance(),
            'root_causes': self.perform_root_cause_analysis(),
            'improvements': self.suggest_improvements()
        }


# Usage example
if __name__ == "__main__":
    # Initialize process mining
    miner = FXProcessMining("staging/fx_trade_log.csv")

    # Execute full analysis
    miner.preprocess_data()
    miner.discover_process()

    # Generate report
    report = miner.generate_report()

    # Print key findings
    print("=== Process Mining Analysis Report ===")
    print(f"\nConformance Metrics:")
    print(f"Fitness: {report['conformance']['fitness']:.2f}")
    print(f"Completed Traces: {report['conformance']['completed_traces']}")

    print("\nTop Bottlenecks:")
    for bottleneck in report['performance']['bottlenecks'][:3]:
        print(f"- {bottleneck['source']} → {bottleneck['target']}: "
              f"{bottleneck['avg_duration']:.2f} seconds")

    print("\nSuggested Improvements:")
    for improvement in report['improvements']:
        print(f"\n{improvement['type'].title()}:")
        print(f"Issue: {improvement['issue']}")
        print(f"Recommendation: {improvement['recommendation']}")