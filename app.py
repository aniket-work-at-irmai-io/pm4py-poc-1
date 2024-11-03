import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta
import pm4py
import base64
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
import os

# Set up Graphviz path
os.environ["PATH"] += os.pathsep + 'C:/samadhi/technology/Graphviz/bin'

# Set page configuration
st.set_page_config(
    page_title="IRAMI OC Process Mining Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .report-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)


def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'output']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def save_uploaded_file(uploaded_file):
    """Save uploaded file to data directory"""
    create_directories()
    file_path = os.path.join('data', uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def process_mining_analysis(csv_path):
    """Perform process mining analysis similar to test.py"""
    # Read CSV file with semicolon separator
    df = pd.read_csv(csv_path, sep=';')

    # Format dataframe according to PM4Py requirements
    log = pm4py.format_dataframe(
        df,
        case_id='case_id',
        activity_key='activity',
        timestamp_key='timestamp'
    )

    # Convert to event log
    event_log = pm4py.convert_to_event_log(log)

    # Save XES file
    xes_path = os.path.join('data', 'event_log.xes')
    pm4py.write_xes(event_log, xes_path)

    # Process Mining with Inductive Miner
    process_tree = inductive_miner.apply(event_log)
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(process_tree)

    # Generate and save visualizations
    pn_gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    pn_visualizer.save(pn_gviz, "output/fx_trade_petri_net.png")

    pt_gviz = pt_visualizer.apply(process_tree)
    pt_visualizer.save(pt_gviz, "output/fx_trade_process_tree.png")

    # Convert to BPMN and save
    bpmn_graph = pm4py.convert_to_bpmn(process_tree)
    bpmn_gviz = bpmn_visualizer.apply(bpmn_graph)
    bpmn_visualizer.save(bpmn_gviz, "output/fx_trade_bpmn.png")

    # Calculate metrics
    fitness = replay_fitness.apply(event_log, net, initial_marking, final_marking)
    precision = precision_evaluator.apply(event_log, net, initial_marking, final_marking)

    # Get start and end activities
    start_activities = pm4py.get_start_activities(event_log)
    end_activities = pm4py.get_end_activities(event_log)

    return fitness, precision, start_activities, end_activities


def main():
    # Header
    st.title("🔄 IRAMI OC Process Mining Dashboard")
    st.markdown("### Professional Process Mining Analysis Tool")

    # Instructions
    st.info(
        "Please install Graphviz and ensure it's in your system PATH (C:/Program Files/Graphviz/bin) before using this tool.")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file (semicolon separated)", type=['csv'])

    if uploaded_file is not None:
        try:
            # Save and process file
            with st.spinner('Processing your data...'):
                file_path = save_uploaded_file(uploaded_file)
                fitness, precision, start_activities, end_activities = process_mining_analysis(file_path)

            # Success message
            st.success('Analysis completed successfully!')

            # Display process information
            st.subheader("Process Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Start Activities:**")
                for activity, count in start_activities.items():
                    st.write(f"- {activity}: {count}")
            with col2:
                st.write("**End Activities:**")
                for activity, count in end_activities.items():
                    st.write(f"- {activity}: {count}")

            # Display metrics
            st.subheader("Process Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fitness Score", f"{fitness['average_trace_fitness']:.2f}")
            with col2:
                st.metric("Precision Score", f"{precision:.2f}")

            # Display visualizations
            st.subheader("Process Visualizations")
            tabs = st.tabs(["BPMN Diagram", "Petri Net", "Process Tree"])

            with tabs[0]:
                st.image("output/fx_trade_bpmn.png", use_column_width=True)
            with tabs[1]:
                st.image("output/fx_trade_petri_net.png", use_column_width=True)
            with tabs[2]:
                st.image("output/fx_trade_process_tree.png", use_column_width=True)

            # Download section
            st.subheader("Download Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                with open("output/fx_trade_bpmn.png", "rb") as file:
                    st.download_button(
                        label="Download BPMN Diagram",
                        data=file,
                        file_name="fx_trade_bpmn.png",
                        mime="image/png"
                    )
            with col2:
                with open("output/fx_trade_petri_net.png", "rb") as file:
                    st.download_button(
                        label="Download Petri Net",
                        data=file,
                        file_name="fx_trade_petri_net.png",
                        mime="image/png"
                    )
            with col3:
                with open("output/fx_trade_process_tree.png", "rb") as file:
                    st.download_button(
                        label="Download Process Tree",
                        data=file,
                        file_name="fx_trade_process_tree.png",
                        mime="image/png"
                    )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please ensure:")
            st.write("1. Your CSV file has the required columns: case_id, activity, timestamp")
            st.write("2. Graphviz is installed and in your system PATH")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>© 2024 IRAMI OC Process Mining. All rights reserved.</p>
            <p>Professional Process Mining Solutions</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()