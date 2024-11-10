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

from risk_analysis import ProcessPathAnalyzer, EventLogAnalyzer, EnhancedRiskAnalyzer

# Set up Graphviz path
os.environ["PATH"] += os.pathsep + 'C:/samadhi/technology/Graphviz/bin'


import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_mining.log'),
        logging.StreamHandler()
    ]
)

# Set page configuration
st.set_page_config(
    page_title="IRMAI Process Analytics",
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

    # Store event_log for risk analysis
    return fitness, precision, start_activities, end_activities, bpmn_graph, event_log


def visualize_results(bpmn_graph, risk_assessment_results):
    # Create a copy of the BPMN graph to modify
    annotated_graph = bpmn_graph.copy()

    # Add risk information to the graph
    for result in risk_assessment_results:
        node_id = result['failure_mode'].split()[-1]
        node = annotated_graph.get_node_by_id(node_id)
        if node:
            # Modify node properties to include RPN
            node.set_label(f"{node.get_label()}\nRPN: {result['rpn']}")

    # Use pm4py's visualization function
    pm4py.view_bpmn(annotated_graph)


def process_mining_with_risk_assessment(event_log, bpmn_graph):
    """Main function to perform risk assessment on process model"""
    try:
        # Initialize risk analyzer with proper parameters
        risk_analyzer = EnhancedRiskAnalyzer(
            event_log=event_log,
            bpmn_graph=bpmn_graph
        )

        # Get risk assessment results
        risk_assessment_results = risk_analyzer.analyze_risks()

        # Get process metrics
        process_metrics = risk_analyzer.path_analyzer.process_metrics

        return bpmn_graph, risk_assessment_results, process_metrics

    except Exception as e:
        print(f"Error in risk assessment: {str(e)}")
        raise


def visualize_risk_distribution(risk_assessment_results):
    """Create visualization of risk distribution"""
    import plotly.graph_objects as go

    # Extract data
    activities = [r['failure_mode'] for r in risk_assessment_results]
    rpn_values = [r['rpn'] for r in risk_assessment_results]
    severities = [r['severity'] for r in risk_assessment_results]
    likelihoods = [r['likelihood'] for r in risk_assessment_results]

    # Create bubble chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=severities,
        y=likelihoods,
        mode='markers',
        marker=dict(
            size=[r['rpn'] * 5 for r in risk_assessment_results],
            color=rpn_values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="RPN")
        ),
        text=activities,
        hovertemplate=
        "<b>Activity:</b> %{text}<br>" +
        "<b>Severity:</b> %{x:.2f}<br>" +
        "<b>Likelihood:</b> %{y:.2f}<br>" +
        "<b>RPN:</b> %{marker.color:.2f}<br>"
    ))

    fig.update_layout(
        title="Risk Distribution Matrix",
        xaxis_title="Severity",
        yaxis_title="Likelihood",
        showlegend=False
    )

    return fig

def main():
    # Header
    st.title("🔄 IRMAI Process Analytics")
    st.markdown("### Process Analytics Tool")

    # Instructions
    st.info(
        "Please install Graphviz and ensure it's in your system PATH (C:/Program Files/Graphviz/bin) before using this tool."
    )

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file (semicolon separated)", type=['csv'])

    if uploaded_file is not None:
        try:
            # Save and process file
            with st.spinner('Processing your data...'):
                file_path = save_uploaded_file(uploaded_file)

                # Process mining analysis - now returns event_log as well
                fitness, precision, start_activities, end_activities, bpmn_graph, event_log = process_mining_analysis(
                    file_path)

            # Success message
            st.success('Analysis completed successfully!')

            # Create tabs for different analysis sections
            tabs = st.tabs(["Process Analysis", "Risk Assessment", "Visualizations", "Downloads"])

            # Process Analysis Tab
            with tabs[0]:
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

                st.subheader("Process Metrics")
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Fitness Score", f"{fitness['average_trace_fitness']:.2f}")
                with metrics_col2:
                    st.metric("Precision Score", f"{precision:.2f}")

                    # Risk Assessment Tab
                    with tabs[1]:
                        st.subheader("Risk Analysis")
                        with st.spinner('Performing risk assessment...'):
                            try:
                                # Perform risk assessment with new analyzers
                                bpmn_graph, risk_assessment_results, process_metrics = process_mining_with_risk_assessment(
                                    event_log, bpmn_graph
                                )

                                # Display process metrics
                                st.write("### Process Metrics")
                                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                with metrics_col1:
                                    st.metric("Total Activities", process_metrics['tasks'])
                                with metrics_col2:
                                    st.metric("Decision Points", process_metrics['decision_points'])
                                with metrics_col3:
                                    st.metric("Process Complexity",
                                              f"{process_metrics['process_complexity']:.2f}")

                                if not risk_assessment_results:
                                    st.warning("No risks identified in the process.")
                                else:
                                    # Display risk assessment results
                                    st.write("### Risk Assessment Results")
                                    for risk in risk_assessment_results:
                                        with st.expander(
                                                f"Risk: {risk['failure_mode']} (RPN: {risk['rpn']})"
                                        ):
                                            cols = st.columns(3)
                                            with cols[0]:
                                                st.metric("Severity", f"{risk['severity']:.2f}")
                                            with cols[1]:
                                                st.metric("Likelihood", f"{risk['likelihood']:.2f}")
                                            with cols[2]:
                                                st.metric("Detectability", f"{risk['detectability']:.2f}")

                                            st.write("#### Structural Details")
                                            st.json(risk['structural_details'])

                                            if 'recommendations' in risk and risk['recommendations']:
                                                st.write("#### Recommendations")
                                                for rec in risk['recommendations']:
                                                    st.write(f"- {rec}")

                                    # Add visualization of risk distribution
                                    st.write("### Risk Distribution")
                                    fig = visualize_risk_distribution(risk_assessment_results)
                                    st.plotly_chart(fig)

                            except Exception as e:
                                st.error("Error in risk assessment:")
                                st.error(str(e))
                                st.write("Please check your event log format and ensure all required data is present.")

            # Visualizations Tab
            with tabs[2]:
                st.subheader("Process Visualizations")
                viz_tabs = st.tabs(["BPMN Diagram", "Petri Net", "Process Tree"])

                with viz_tabs[0]:
                    st.image("output/fx_trade_bpmn.png", use_column_width=True)
                with viz_tabs[1]:
                    st.image("output/fx_trade_petri_net.png", use_column_width=True)
                with viz_tabs[2]:
                    st.image("output/fx_trade_process_tree.png", use_column_width=True)

            # Downloads Tab
            with tabs[3]:
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

                # Add download button for risk assessment report
                if 'risk_assessment_results' in locals():
                    st.write("### Risk Assessment Report")
                    report_csv = generate_risk_report_csv(
                        risk_assessment_results)  # You'll need to implement this function
                    st.download_button(
                        label="Download Risk Assessment Report (CSV)",
                        data=report_csv,
                        file_name="risk_assessment_report.csv",
                        mime="text/csv"
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
            <p>© 2024 IRMAI OC Process Mining. All rights reserved.</p>
            <p>Professional Process Mining Solutions</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# Helper function for risk heatmap
def generate_risk_heatmap(risk_assessment_results):
    """Generate a plotly heatmap visualization of risks"""
    import plotly.graph_objects as go

    # Extract data for heatmap
    activities = [r['failure_mode'] for r in risk_assessment_results]
    severity = [r['severity'] for r in risk_assessment_results]
    likelihood = [r['likelihood'] for r in risk_assessment_results]
    rpn_values = [r['rpn'] for r in risk_assessment_results]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[rpn_values],
        x=activities,
        y=['RPN'],
        colorscale='Reds',
        text=[[f"S:{s}<br>L:{l}<br>RPN:{r}" for s, l, r in zip(severity, likelihood, rpn_values)]],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Risk Priority Number")
    ))

    # Update layout
    fig.update_layout(
        title="Risk Priority Number (RPN) Heatmap",
        xaxis_title="Failure Modes",
        yaxis_title="Risk Metric",
        height=400
    )

    return fig


# Helper function for generating risk report CSV
def generate_risk_report_csv(risk_assessment_results):
    """Generate a CSV report of risk assessment results"""
    import io
    import csv

    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(['Failure Mode', 'Severity', 'Likelihood', 'Detectability', 'RPN', 'Recommendations'])

    # Write data
    for result in risk_assessment_results:
        writer.writerow([
            result['failure_mode'],
            result['severity'],
            result['likelihood'],
            result['detectability'],
            result['rpn'],
            '; '.join(result['recommendations'])
        ])

    return output.getvalue()






if __name__ == "__main__":
    main()