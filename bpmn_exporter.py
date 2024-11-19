import pm4py
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
import os
from datetime import datetime
import pandas as pd
from typing import Tuple, Optional
import logging
from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter
from pm4py.objects.bpmn.obj import BPMN
from pm4py.algo.discovery.inductive import algorithm as inductive_miner


class BPMNExporter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('process_mining.log'),
                logging.StreamHandler()
            ]
        )

    def create_and_export_bpmn(self, event_log, output_path: str) -> Tuple[BPMN, str]:
        """
        Creates BPMN model from event log and exports to XML file

        Args:
            event_log: PM4Py event log object
            output_path: Path to save the BPMN XML file

        Returns:
            Tuple of (BPMN object, path to saved file)
        """
        try:
            self.logger.info("Starting BPMN creation process")

            # Apply Inductive Miner to get process tree
            # Using the correct method name for recent PM4Py versions
            process_tree = inductive_miner.apply(event_log)
            self.logger.info("Process tree created successfully")

            # Convert process tree to BPMN
            bpmn_graph = pm4py.convert_to_bpmn(process_tree)
            self.logger.info("BPMN graph created successfully")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Export BPMN to XML file
            bpmn_exporter.apply(
                bpmn_graph,
                output_path
            )
            self.logger.info(f"BPMN XML exported successfully to {output_path}")

            return bpmn_graph, output_path

        except Exception as e:
            self.logger.error(f"Error in BPMN creation/export: {str(e)}")
            raise


def generate_bpmn_from_csv(csv_path: str, output_dir: str = "staging") -> Optional[str]:
    """
    Generates BPMN model from CSV file and exports it

    Args:
        csv_path: Path to input CSV file
        output_dir: Directory to save output files

    Returns:
        Path to generated BPMN file or None if failed
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path, sep=';')

        # Format dataframe for PM4Py
        log = pm4py.format_dataframe(
            df,
            case_id='case_id',
            activity_key='activity',
            timestamp_key='timestamp'
        )

        # Convert to event log
        event_log = pm4py.convert_to_event_log(log)

        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bpmn_path = os.path.join(output_dir, f"process_model_{timestamp}.bpmn")

        # Create and export BPMN
        exporter = BPMNExporter()
        bpmn_graph, saved_path = exporter.create_and_export_bpmn(event_log, bpmn_path)

        return saved_path

    except Exception as e:
        logging.error(f"Error generating BPMN from CSV: {str(e)}")
        return None


def main():
    # Create necessary directories
    os.makedirs("staging", exist_ok=True)

    # Generate BPMN from example data
    input_csv = "staging/fx_trade_log.csv.bk"  # Make sure this exists
    bpmn_file = generate_bpmn_from_csv(input_csv)

    if bpmn_file:
        print(f"BPMN file generated successfully: {bpmn_file}")
    else:
        print("Failed to generate BPMN file")


if __name__ == "__main__":
    main()