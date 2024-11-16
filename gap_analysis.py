# Remove duplicate imports
import threading
import time
import networkx as nx
from xml.etree import ElementTree as ET
import spacy
import PyPDF2
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import os
import logging
from datetime import datetime
import psutil  # Change from plotly.io._orca import psutil to just import psutil


class ModelManager:
    """Manages spaCy model loading and monitoring"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_loaded = threading.Event()
        self.nlp = None

    def load_model(self) -> None:
        """Load the large spaCy model with progress monitoring"""
        try:
            start_time = time.time()
            start_mem = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

            self.logger.info("Loading en_core_web_lg model...")

            try:
                self.nlp = spacy.load('en_core_web_lg')
            except OSError:
                self.logger.info("Model not found. Downloading en_core_web_lg...")
                os.system('python -m spacy download en_core_web_lg')
                self.nlp = spacy.load('en_core_web_lg')

            end_time = time.time()
            end_mem = psutil.Process().memory_info().rss / (1024 * 1024)

            self.logger.info(
                f"Model loaded in {end_time - start_time:.2f}s. "
                f"Memory usage: {end_mem - start_mem:.1f}MB"
            )

            self.model_loaded.set()

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise


# In gap_analysis.py

class ProcessGuidelinesComparator:
    """Compares actual BPMN process against expected guidelines to identify gaps"""

    def __init__(self):
        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('process_mining.log'),
                logging.StreamHandler()
            ]
        )

        # Initialize NLP model for text analysis
        self.model_manager = ModelManager()
        self.model_manager.load_model()
        self.nlp = self.model_manager.nlp

        # Track actual process components from BPMN
        self.actual_process = {
            'activities': [],  # List of activity nodes
            'sequence_flows': [],  # Process flow connections
            'gateways': [],  # Decision/merge points
            'controls': []  # Control points
        }

        # Track expected components from guidelines
        self.expected_guidelines = {
            'activities': [],  # Required process steps
            'controls': [],  # Required control points
            'risks': [],  # Risk considerations
            'requirements': []  # Other requirements
        }

        # Store gap analysis results
        self.gap_analysis = {
            'missing_activities': [],  # Expected activities not in BPMN
            'missing_controls': [],  # Expected controls not in BPMN
            'sequence_violations': [],  # Flow issues
            'risk_gaps': [],  # Unaddressed risks
            'compliance_gaps': []  # Other gaps
        }

    def analyze_process_flow(self, bpmn_graph):
        """Extract and analyze process flow from BPMN XML"""
        self.logger.info("Starting BPMN process flow analysis...")
        try:
            # Define BPMN namespace
            ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

            # Find process element
            process = bpmn_graph.find('.//bpmn:process', ns)
            if not process:
                raise ValueError("No process found in BPMN file")

            # Extract activities (tasks)
            tasks = process.findall('.//bpmn:task', ns)
            for task in tasks:
                activity = {
                    'id': task.get('id'),
                    'name': task.get('name', ''),
                    'type': 'activity',
                    'incoming': [],
                    'outgoing': []
                }

                # Find incoming/outgoing flows
                for flow in process.findall('.//bpmn:sequenceFlow', ns):
                    if flow.get('targetRef') == activity['id']:
                        activity['incoming'].append(flow.get('sourceRef'))
                    if flow.get('sourceRef') == activity['id']:
                        activity['outgoing'].append(flow.get('targetRef'))

                self.actual_process['activities'].append(activity)

            # Extract gateways
            gateway_types = [
                'exclusiveGateway',
                'parallelGateway',
                'inclusiveGateway'
            ]

            for gateway_type in gateway_types:
                gateways = process.findall(f'.//bpmn:{gateway_type}', ns)
                for gateway in gateways:
                    gateway_obj = {
                        'id': gateway.get('id'),
                        'type': gateway_type,
                        'incoming': [],
                        'outgoing': []
                    }

                    # Find incoming/outgoing flows
                    for flow in process.findall('.//bpmn:sequenceFlow', ns):
                        if flow.get('targetRef') == gateway_obj['id']:
                            gateway_obj['incoming'].append(flow.get('sourceRef'))
                        if flow.get('sourceRef') == gateway_obj['id']:
                            gateway_obj['outgoing'].append(flow.get('targetRef'))

                    self.actual_process['gateways'].append(gateway_obj)

            # Extract sequence flows
            flows = process.findall('.//bpmn:sequenceFlow', ns)
            for flow in flows:
                flow_obj = {
                    'id': flow.get('id'),
                    'source': flow.get('sourceRef'),
                    'target': flow.get('targetRef')
                }
                self.actual_process['sequence_flows'].append(flow_obj)

            # Find control points (boundary events, monitoring tasks etc.)
            control_elements = process.findall('.//bpmn:boundaryEvent', ns) + \
                               process.findall('.//bpmn:intermediateCatchEvent', ns)

            for control in control_elements:
                control_obj = {
                    'id': control.get('id'),
                    'name': control.get('name', ''),
                    'type': 'control',
                    'attached_to': control.get('attachedToRef', '')
                }
                self.actual_process['controls'].append(control_obj)

            self.logger.info(
                f"Process analysis complete. Found:"
                f"\n- {len(self.actual_process['activities'])} activities"
                f"\n- {len(self.actual_process['gateways'])} gateways"
                f"\n- {len(self.actual_process['sequence_flows'])} sequence flows"
                f"\n- {len(self.actual_process['controls'])} control points"
            )

        except Exception as e:
            self.logger.error(f"Error analyzing process flow: {str(e)}")
            raise

    def parse_guidelines(self, pdf_content):
        """Extract expected process elements from guidelines document"""
        self.logger.info("Starting guidelines document analysis...")
        try:
            # Process PDF content with NLP
            doc = self.nlp(pdf_content)

            # Process each sentence
            for sent in doc.sents:
                sent_text = sent.text.lower()

                # Look for activity descriptions
                if any(word in sent_text for word in ['must', 'should', 'shall', 'required']):
                    activity = self._extract_activity_requirement(sent)
                    if activity:
                        self.expected_guidelines['activities'].append(activity)

                # Look for control requirements
                if any(word in sent_text for word in ['control', 'monitor', 'verify', 'validate']):
                    control = self._extract_control_requirement(sent)
                    if control:
                        self.expected_guidelines['controls'].append(control)

                # Look for risk requirements
                if any(word in sent_text for word in ['risk', 'exposure', 'threat', 'vulnerability']):
                    risk = self._extract_risk_requirement(sent)
                    if risk:
                        self.expected_guidelines['risks'].append(risk)

            self.logger.info(f"Guidelines analysis complete. Found {len(self.expected_guidelines['activities'])} "
                             f"activities, {len(self.expected_guidelines['controls'])} controls, "
                             f"{len(self.expected_guidelines['risks'])} risks")

        except Exception as e:
            self.logger.error(f"Error parsing guidelines: {str(e)}")
            raise

    def _extract_activity_requirement(self, sentence):
        """Extract activity details from sentence"""
        try:
            # Extract main verb and objects
            main_verb = None
            objects = []

            for token in sentence:
                if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                    main_verb = token.lemma_
                elif token.dep_ in ['dobj', 'pobj'] and not token.is_stop:
                    objects.append(token.text)

            if main_verb:
                return {
                    'name': f"{main_verb} {' '.join(objects)}",
                    'verb': main_verb,
                    'objects': objects,
                    'original_text': sentence.text
                }
            return None

        except Exception as e:
            self.logger.error(f"Error extracting activity: {str(e)}")
            return None

    def _extract_control_requirement(self, sentence):
        """Extract control requirement details from sentence"""
        try:
            control_type = None
            description = []

            for token in sentence:
                if token.text.lower() in ['control', 'monitor', 'verify', 'validate']:
                    control_type = token.text
                elif not token.is_stop:
                    description.append(token.text)

            if control_type:
                return {
                    'type': control_type,
                    'name': ' '.join(description),
                    'original_text': sentence.text
                }
            return None

        except Exception as e:
            self.logger.error(f"Error extracting control: {str(e)}")
            return None

    def _extract_risk_requirement(self, sentence):
        """Extract risk requirement details from sentence"""
        try:
            risk_type = None
            description = []
            severity = 'medium'  # Default

            for token in sentence:
                if token.text.lower() in ['risk', 'threat', 'vulnerability']:
                    risk_type = token.text
                elif token.text.lower() in ['high', 'critical', 'severe']:
                    severity = 'high'
                elif token.text.lower() in ['low', 'minor']:
                    severity = 'low'
                elif not token.is_stop:
                    description.append(token.text)

            if risk_type:
                return {
                    'type': risk_type,
                    'description': ' '.join(description),
                    'severity': severity,
                    'original_text': sentence.text
                }
            return None

        except Exception as e:
            self.logger.error(f"Error extracting risk: {str(e)}")
            return None

    def perform_gap_analysis(self):
        """Compare actual process against guidelines"""
        self.logger.info("Starting gap analysis...")
        try:
            # Compare activities
            actual_activities = set(a['name'].lower() for a in self.actual_process['activities'])
            expected_activities = set(a['name'].lower() for a in self.expected_guidelines['activities'])

            # Find missing activities
            self.gap_analysis['missing_activities'] = [
                activity for activity in self.expected_guidelines['activities']
                if activity['name'].lower() not in actual_activities
            ]

            # Compare controls
            actual_controls = set(c.get('name', '').lower() for c in self.actual_process.get('controls', []))
            expected_controls = set(c['name'].lower() for c in self.expected_guidelines['controls'])

            # Find missing controls
            self.gap_analysis['missing_controls'] = [
                control for control in self.expected_guidelines['controls']
                if control['name'].lower() not in actual_controls
            ]

            # Analyze sequence compliance
            self._analyze_sequence_compliance()

            # Analyze risk coverage
            self._analyze_risk_coverage()

            # Generate and return report
            return self.generate_gap_report()

        except Exception as e:
            self.logger.error(f"Error performing gap analysis: {str(e)}")
            raise

    def _analyze_sequence_compliance(self):
        """Check process flow against expected sequences"""
        try:
            # Look for required sequences in guidelines
            for activity in self.expected_guidelines['activities']:
                # Find corresponding BPMN activity
                bpmn_activity = self._find_matching_activity(activity['name'])
                if not bpmn_activity:
                    continue

                # Check if sequence is correct
                if not self._verify_activity_sequence(bpmn_activity, activity):
                    self.gap_analysis['sequence_violations'].append({
                        'activity': activity['name'],
                        'issue': 'Invalid sequence',
                        'details': f"Activity {activity['name']} is not in correct sequence"
                    })

        except Exception as e:
            self.logger.error(f"Error analyzing sequence compliance: {str(e)}")
            raise

    def _analyze_risk_coverage(self):
        """Check if process addresses identified risks"""
        try:
            for risk in self.expected_guidelines['risks']:
                # Check if risk is addressed by controls
                if not self._find_risk_controls(risk):
                    self.gap_analysis['risk_gaps'].append({
                        'risk': risk['description'],
                        'severity': risk['severity'],
                        'recommendation': self._generate_risk_recommendation(risk)
                    })

        except Exception as e:
            self.logger.error(f"Error analyzing risk coverage: {str(e)}")
            raise

    def _find_matching_activity(self, activity_name):
        """Find BPMN activity matching guideline activity"""
        activity_name = activity_name.lower()
        for activity in self.actual_process['activities']:
            if activity['name'].lower() == activity_name:
                return activity
        return None

    def _verify_activity_sequence(self, bpmn_activity, guideline_activity):
        """Verify activity follows correct sequence"""
        # Get expected previous/next activities
        prev_activities = self._get_previous_activities(bpmn_activity)
        next_activities = self._get_next_activities(bpmn_activity)

        # Compare with guidelines
        return self._sequence_matches_guidelines(
            prev_activities,
            next_activities,
            guideline_activity
        )

    def _find_risk_controls(self, risk):
        """Find controls addressing a specific risk"""
        risk_words = set(risk['description'].lower().split())

        for control in self.actual_process.get('controls', []):
            control_words = set(control.get('name', '').lower().split())
            if risk_words & control_words:  # If words overlap
                return True
        return False

    def _generate_risk_recommendation(self, risk):
        """Generate recommendation for unaddressed risk"""
        if risk['severity'] == 'high':
            return f"HIGH PRIORITY: Implement control measures for {risk['description']}"
        elif risk['severity'] == 'medium':
            return f"Add controls to address {risk['description']}"
        else:
            return f"Consider adding controls for {risk['description']}"

    def generate_gap_report(self):
        """Generate detailed gap analysis report"""
        try:
            report = []
            report.append("=== FX Trading Process Gap Analysis Report ===")
            report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # Process Overview
            report.append("1. Process Overview")
            report.append(f"   Actual Activities: {len(self.actual_process['activities'])}")
            report.append(f"   Expected Activities: {len(self.expected_guidelines['activities'])}")
            report.append(f"   Control Points: {len(self.actual_process.get('controls', []))}")
            report.append(f"   Expected Controls: {len(self.expected_guidelines['controls'])}\n")

            # Missing Activities
            report.append("2. Missing Required Activities")
            if self.gap_analysis['missing_activities']:
                for activity in self.gap_analysis['missing_activities']:
                    report.append(f"   - Activity: {activity['name']}")
                    report.append(f"     Context: {activity['original_text']}\n")
            else:
                report.append("   No missing activities identified.\n")

            # Missing Controls
            report.append("3. Missing Control Requirements")
            if self.gap_analysis['missing_controls']:
                for control in self.gap_analysis['missing_controls']:
                    report.append(f"   - Control: {control['name']}")
                    report.append(f"     Type: {control['type']}")
                    report.append(f"     Context: {control['original_text']}\n")
            else:
                report.append("   No missing controls identified.\n")

            # Sequence Issues
            report.append("4. Process Flow Issues")
            if self.gap_analysis['sequence_violations']:
                for violation in self.gap_analysis['sequence_violations']:
                    report.append(f"   - Activity: {violation['activity']}")
                    report.append(f"     Issue: {violation['issue']}")
                    report.append(f"     Details: {violation['details']}\n")
            else:
                report.append("   No sequence violations identified.\n")

            # Risk Coverage
            report.append("5. Risk Coverage Gaps")
            if self.gap_analysis['risk_gaps']:
                for gap in self.gap_analysis['risk_gaps']:
                    report.append(f"   Risk: {gap['risk']}")
                    report.append(f"   Severity: {gap['severity']}")
                    report.append(f"   Recommendation: {gap['recommendation']}\n")
            else:
                report.append("   No risk coverage gaps identified.\n")

            # Compliance Summary
            report.append("6. Compliance Analysis")
            # Calculate compliance percentages
            total_expected = len(self.expected_guidelines['activities']) + len(self.expected_guidelines['controls'])
            total_missing = len(self.gap_analysis['missing_activities']) + len(
                self.gap_analysis['missing_controls'])
            compliance_rate = ((total_expected - total_missing) / total_expected * 100) if total_expected > 0 else 0

            report.append(f"   Overall Compliance Rate: {compliance_rate:.1f}%")
            report.append(f"   Activities Compliance: {self._calculate_activity_compliance():.1f}%")
            report.append(f"   Controls Compliance: {self._calculate_control_compliance():.1f}%")

            # Detailed Findings
            report.append("\n7. Detailed Findings")
            findings = self._generate_detailed_findings()
            report.extend(findings)

            # Recommendations
            report.append("\n8. Recommendations")
            recommendations = self._generate_recommendations()
            report.extend(recommendations)

            # Save report to file
            report_text = '\n'.join(report)
            report_path = "gap_analysis_report.txt"
            with open(report_path, 'w') as f:
                f.write(report_text)

            self.logger.info(f"Gap analysis report generated and saved to: {report_path}")
            return report_text

        except Exception as e:
            self.logger.error(f"Error generating gap report: {str(e)}")
            raise

    def _calculate_activity_compliance(self) -> float:
        """Calculate activity compliance percentage"""
        total_expected = len(self.expected_guidelines['activities'])
        missing = len(self.gap_analysis['missing_activities'])
        return ((total_expected - missing) / total_expected * 100) if total_expected > 0 else 0

    def _calculate_control_compliance(self) -> float:
        """Calculate control compliance percentage"""
        total_expected = len(self.expected_guidelines['controls'])
        missing = len(self.gap_analysis['missing_controls'])
        return ((total_expected - missing) / total_expected * 100) if total_expected > 0 else 0

    def _generate_detailed_findings(self) -> List[str]:
        """Generate detailed analysis findings"""
        findings = []

        # Analyze critical path activities
        critical_activities = self._identify_critical_activities()
        if critical_activities:
            findings.append("Critical Path Activities:")
            for activity in critical_activities:
                findings.append(f"   - {activity['name']}")
                if activity['name'].lower() in [a['name'].lower() for a in self.gap_analysis['missing_activities']]:
                    findings.append("     WARNING: Critical activity missing from actual process")

        # Analyze control effectiveness
        control_findings = self._analyze_control_effectiveness()
        if control_findings:
            findings.append("\nControl Effectiveness:")
            findings.extend(control_findings)

        # Analyze risk distribution
        risk_findings = self._analyze_risk_distribution()
        if risk_findings:
            findings.append("\nRisk Distribution:")
            findings.extend(risk_findings)

        return findings

    def _generate_recommendations(self) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []

        # High priority recommendations
        high_priority = []
        for gap in self.gap_analysis['risk_gaps']:
            if gap['severity'] == 'high':
                high_priority.append(f"   - {gap['recommendation']}")

        if high_priority:
            recommendations.append("High Priority Actions:")
            recommendations.extend(high_priority)

        # Process improvements
        process_recs = []
        for violation in self.gap_analysis['sequence_violations']:
            process_recs.append(f"   - Correct sequence for {violation['activity']}: {violation['details']}")

        if process_recs:
            recommendations.append("\nProcess Improvements:")
            recommendations.extend(process_recs)

        # Control enhancements
        control_recs = []
        for control in self.gap_analysis['missing_controls']:
            control_recs.append(f"   - Implement {control['type']} control: {control['name']}")

        if control_recs:
            recommendations.append("\nControl Enhancements:")
            recommendations.extend(control_recs)

        return recommendations

    def _identify_critical_activities(self) -> List[Dict]:
        """Identify critical path activities"""
        return [
            activity for activity in self.actual_process['activities']
            if self._is_critical_activity(activity)
        ]

    def _is_critical_activity(self, activity: Dict) -> bool:
        """Determine if activity is on critical path"""
        # Activity is critical if it has multiple incoming/outgoing flows
        # or is referenced in high-severity risks
        incoming_count = len(activity.get('incoming', []))
        outgoing_count = len(activity.get('outgoing', []))

        if incoming_count > 1 or outgoing_count > 1:
            return True

        # Check if activity is mentioned in high-severity risks
        activity_name = activity['name'].lower()
        for risk in self.expected_guidelines['risks']:
            if risk['severity'] == 'high' and activity_name in risk['description'].lower():
                return True

        return False

    def _analyze_control_effectiveness(self) -> List[str]:
        """Analyze effectiveness of existing controls"""
        findings = []
        actual_controls = self.actual_process.get('controls', [])

        for control in actual_controls:
            # Analyze control coverage
            covered_risks = self._find_covered_risks(control)
            findings.append(f"   Control: {control.get('name', 'Unnamed control')}")
            findings.append(f"     Covers {len(covered_risks)} identified risks")

            # Assess control placement
            placement_issues = self._assess_control_placement(control)
            if placement_issues:
                findings.append(f"     Placement issues: {', '.join(placement_issues)}")

        return findings

    def _analyze_risk_distribution(self) -> List[str]:
        """Analyze distribution of risks across process"""
        findings = []
        risk_counts = defaultdict(int)

        for risk in self.expected_guidelines['risks']:
            risk_counts[risk['severity']] += 1

        findings.append(f"   High severity risks: {risk_counts['high']}")
        findings.append(f"   Medium severity risks: {risk_counts['medium']}")
        findings.append(f"   Low severity risks: {risk_counts['low']}")

        # Analyze risk clustering
        clusters = self._identify_risk_clusters()
        if clusters:
            findings.append("\n   Risk clusters identified:")
            for cluster in clusters:
                findings.append(f"     - {cluster['description']}")

        return findings

    def _find_covered_risks(self, control: Dict) -> List[Dict]:
        """Find risks addressed by a specific control"""
        covered_risks = []
        control_text = control.get('name', '').lower()

        for risk in self.expected_guidelines['risks']:
            if any(word in control_text for word in risk['description'].lower().split()):
                covered_risks.append(risk)

        return covered_risks

    def _assess_control_placement(self, control: Dict) -> List[str]:
        """Assess if control is properly placed in process"""
        issues = []

        # Check if control has proper incoming/outgoing flows
        if not control.get('incoming'):
            issues.append("No incoming flows")
        if not control.get('outgoing'):
            issues.append("No outgoing flows")

        # Check if control is in appropriate sequence
        if not self._verify_control_sequence(control):
            issues.append("Incorrect sequence")

        return issues

    def _identify_risk_clusters(self) -> List[Dict]:
        """Identify clusters of related risks"""
        clusters = []
        processed_risks = set()

        for risk in self.expected_guidelines['risks']:
            if risk['description'] in processed_risks:
                continue

            # Find related risks
            related = self._find_related_risks(risk)
            if len(related) > 1:  # If we found a cluster
                clusters.append({
                    'description': f"Cluster around {risk['description'][:50]}...",
                    'risks': related,
                    'severity': max(r['severity'] for r in related)
                })

            processed_risks.update(r['description'] for r in related)

        return clusters

    def _find_related_risks(self, risk: Dict) -> List[Dict]:
        """Find risks related to given risk"""
        related = [risk]
        risk_words = set(risk['description'].lower().split())

        for other_risk in self.expected_guidelines['risks']:
            if other_risk['description'] == risk['description']:
                continue

            other_words = set(other_risk['description'].lower().split())
            # If significant word overlap, consider related
            if len(risk_words & other_words) / len(risk_words | other_words) > 0.3:
                related.append(other_risk)

        return related


def main():
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('process_mining.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)

        logger.info("Starting gap analysis process")

        # Create staging directory
        os.makedirs("staging", exist_ok=True)
        logger.info("Created staging directory")

        # Define input files
        bpmn_path = "staging/fx_trade_process.bpmn"
        guidelines_path = "staging/tradingguidelinesNov2010.pdf"

        # Check input files exist
        logger.info("Checking input files...")
        for filepath in [bpmn_path, guidelines_path]:
            if not os.path.exists(filepath):
                logger.error(f"Required file not found: {filepath}")
                raise FileNotFoundError(f"Required file not found: {filepath}")
            logger.info(f"Found input file: {filepath}")

        # Initialize comparator
        logger.info("Initializing ProcessGuidelinesComparator")
        comparator = ProcessGuidelinesComparator()

        # Parse BPMN
        logger.info(f"Parsing BPMN from: {bpmn_path}")
        tree = ET.parse(bpmn_path)
        root = tree.getroot()
        bpmn_graph = root  # Since we already have the BPMN XML parsed
        comparator.analyze_process_flow(bpmn_graph)
        logger.info("BPMN analysis complete")

        # Parse guidelines
        logger.info(f"Starting guidelines parsing from: {guidelines_path}")
        with open(guidelines_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
        comparator.parse_guidelines(text)
        logger.info("Guidelines parsing complete")

        # Perform gap analysis and generate report
        logger.info("Starting gap analysis...")
        report = comparator.perform_gap_analysis()

        # Print report summary
        print("\nGap Analysis Results:")
        print("=====================")
        print(f"\nFull report saved to: gap_analysis_report.txt")

        # Print key findings
        print("\nKey Findings:")
        if comparator.gap_analysis['missing_activities']:
            print(f"\nMissing Activities: {len(comparator.gap_analysis['missing_activities'])}")
            for activity in comparator.gap_analysis['missing_activities'][:3]:  # Show first 3
                print(f"- {activity['name']}")

        if comparator.gap_analysis['missing_controls']:
            print(f"\nMissing Controls: {len(comparator.gap_analysis['missing_controls'])}")
            for control in comparator.gap_analysis['missing_controls'][:3]:  # Show first 3
                print(f"- {control['name']}")

        if comparator.gap_analysis['risk_gaps']:
            print(f"\nRisk Gaps: {len(comparator.gap_analysis['risk_gaps'])}")
            for risk in comparator.gap_analysis['risk_gaps'][:3]:  # Show first 3
                print(f"- {risk['risk']} (Severity: {risk['severity']})")

    except Exception as e:
        logger.error(f"Fatal error in analysis: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        print("Check process_mining.log for details")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

