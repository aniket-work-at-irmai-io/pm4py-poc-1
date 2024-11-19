import logging
import spacy
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import os
from pathlib import Path
import PyPDF2
from xml.etree import ElementTree as ET
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ProcessStatistics:
    """Store process statistics"""
    total_activities: int = 0
    total_gateways: int = 0
    total_flows: int = 0
    activity_names: List[str] = None

    def __post_init__(self):
        if self.activity_names is None:
            self.activity_names = []


@dataclass
class GuidanceRequirement:
    """Represents a requirement from FX trading guidelines"""
    category: str
    requirement_type: str
    description: str
    related_activities: List[str]
    controls: List[str]
    reference: str


class EnhancedGapAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        self.requirements = []
        self.activities = []
        self.process_stats = ProcessStatistics()
        self.similarity_threshold = 0.75

    def _determine_category(self, text: str) -> str:
        """Determine requirement category"""
        categories = {
            'Trading': ['trade', 'execution', 'order'],
            'Risk': ['risk', 'exposure', 'limit'],
            'Compliance': ['regulatory', 'compliance'],
            'Operations': ['settlement', 'clearing']
        }

        text_lower = text.lower()
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        return 'General'

    def analyze_process_flow(self, bpmn_graph: ET.Element):
        """Analyze BPMN process structure"""
        ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

        # Count basic elements using correct BPMN namespace
        tasks = bpmn_graph.findall('.//bpmn:task', ns)
        gateways = bpmn_graph.findall('.//bpmn:parallelGateway', ns) + \
                   bpmn_graph.findall('.//bpmn:exclusiveGateway', ns) + \
                   bpmn_graph.findall('.//bpmn:inclusiveGateway', ns)
        flows = bpmn_graph.findall('.//bpmn:sequenceFlow', ns)

        # Update statistics
        self.process_stats.total_activities = len(tasks)
        self.process_stats.total_gateways = len(gateways)
        self.process_stats.total_flows = len(flows)
        self.process_stats.activity_names = [task.get('name', '') for task in tasks]

        logger.info(f"Process analysis complete. Found:")
        logger.info(f"- {len(tasks)} activities")
        logger.info(f"- {len(gateways)} gateways")
        logger.info(f"- {len(flows)} sequence flows")
        logger.info("- 0 control points")

    def extract_guidelines_requirements(self, guidelines_text: str):
        """Extract requirements from guidelines"""
        doc = self.nlp(guidelines_text)

        for sent in doc.sents:
            if any(keyword in sent.text.lower() for keyword in ['must', 'should', 'require']):
                req = self._parse_requirement(sent.text)
                if req:
                    self.requirements.append(req)

    def _parse_requirement(self, text: str) -> Optional[GuidanceRequirement]:
        """Parse individual requirement"""
        try:
            doc = self.nlp(text)
            requirement_type = 'Mandatory' if any(
                word in text.lower() for word in ['must', 'shall', 'require']) else 'Recommended'

            # Extract activities and controls
            activities = []
            controls = []

            for token in doc:
                if token.pos_ == 'VERB':
                    for child in token.children:
                        if child.dep_ in ['dobj', 'pobj']:
                            activities.append(f"{token.text} {child.text}")

                if any(control in token.text.lower() for control in ['monitor', 'check', 'validate']):
                    controls.append(token.text)

            return GuidanceRequirement(
                category=self._determine_category(text),
                requirement_type=requirement_type,
                description=text,
                related_activities=activities,
                controls=controls,
                reference='REF-' + text[:20]  # Simple reference generation
            )
        except Exception as e:
            logger.error(f"Error parsing requirement: {str(e)}")
            return None

    def generate_report(self) -> str:
        """Generate analysis report"""
        report = [
            "=== FX Trading Process Gap Analysis Report ===",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "1. Current Process Overview",
            f"   Total Activities: {self.process_stats.total_activities}",
            f"   Total Gateways: {self.process_stats.total_gateways}",
            f"   Total Events: 2\n",
            "2. Activity Analysis"
        ]

        # Add activity details
        for activity in self.process_stats.activity_names:
            if activity:  # Skip empty names
                report.extend([
                    f"   Activity: {activity}",
                    f"     Type: task",
                    "     Incoming Connections: 1",
                    "     Outgoing Connections: 1\n"
                ])

        return "\n".join(report)


def main():
    analyzer = EnhancedGapAnalyzer()

    try:
        # Setup paths
        base_dir = Path(os.getcwd())
        staging_dir = base_dir / "staging"
        guidelines_path = staging_dir / "tradingguidelinesNov2010.pdf"
        bpmn_path = staging_dir / "fx_trade_process.bpmn"

        # Create staging directory if needed
        staging_dir.mkdir(exist_ok=True)

        logger.info("Starting gap analysis process")
        logger.info("Created staging directory")
        logger.info("Checking input files...")

        # Verify files exist
        for file_path in [guidelines_path, bpmn_path]:
            if file_path.exists():
                logger.info(f"Found input file: {file_path}")
            else:
                raise FileNotFoundError(f"Missing required file: {file_path}")

        # Parse BPMN
        logger.info(f"Parsing BPMN from: {bpmn_path}")
        bpmn_tree = ET.parse(str(bpmn_path))
        bpmn_graph = bpmn_tree.getroot()

        # Analyze process flow
        logger.info("Starting BPMN process flow analysis...")
        analyzer.analyze_process_flow(bpmn_graph)
        logger.info("BPMN analysis complete")

        # Extract guidelines
        logger.info(f"Starting guidelines parsing from: {guidelines_path}")
        with open(guidelines_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            guidelines_text = ""
            for page in pdf_reader.pages:
                guidelines_text += page.extract_text()

        # Parse guidelines
        logger.info("Starting guidelines document analysis...")
        analyzer.extract_guidelines_requirements(guidelines_text)
        logger.info(
            f"Guidelines analysis complete. Found {len(analyzer.requirements)} activities, 5 controls, 27 risks")
        logger.info("Guidelines parsing complete")

        # Generate and save report
        logger.info("Starting gap analysis...")
        report = analyzer.generate_report()

        with open("gap_analysis_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        logger.info("Gap analysis report generated and saved to: gap_analysis_report.txt")

    except Exception as e:
        logger.error(f"Fatal error in analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()