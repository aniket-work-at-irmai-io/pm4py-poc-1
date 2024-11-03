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
os.environ["PATH"] += os.pathsep + 'C:/samadhi/technology/Graphviz/bin'



# 1. Generate Synthetic Event Log
def generate_fx_trade_event_log(num_cases=100):
    random.seed(42)
    event_types = ["Trade Initiated", "Trade Executed", "Trade Allocated", "Trade Settled", "Trade Canceled"]
    traders = ["Trader A", "Trader B", "Trader C", "Trader D"]
    currencies = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
    event_log = []

    for case_id in range(1, num_cases + 1):
        num_events = random.randint(2, 5)
        case_events = random.sample(event_types, num_events)
        case_events.sort(key=lambda x: event_types.index(x))

        trader = random.choice(traders)
        currency = random.choice(currencies)
        start_time = datetime.now() + timedelta(days=random.randint(-30, 0))

        for i, event in enumerate(case_events):
            timestamp = start_time + timedelta(minutes=random.randint(10, 60) * i)
            event_log.append({
                "case:concept:name": f"Case_{case_id}",
                "concept:name": event,
                "time:timestamp": timestamp,
                "org:resource": trader,
                "currency_pair": currency
            })

    df = pd.DataFrame(event_log)
    df.sort_values(by=["case:concept:name", "time:timestamp"], inplace=True)
    return df


# Generate the event log
fx_event_log = generate_fx_trade_event_log(100)
print(fx_event_log.head())

event_log_xes = pm4py.convert_to_event_log(fx_event_log)
pm4py.write_xes(event_log_xes, "fx_trade_log.xes")

# 2. Process Mining with Inductive Miner
event_log = pm4py.read_xes("fx_trade_log.xes")
process_tree = inductive_miner.apply(event_log)
net, initial_marking, final_marking = pm4py.convert_to_petri_net(process_tree)

pn_gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.save(pn_gviz, "fx_trade_petri_net.png")

pt_gviz = pt_visualizer.apply(process_tree)
pt_visualizer.save(pt_gviz, "fx_trade_process_tree.png")

# 3. Analysis on Petri Net
fitness = replay_fitness.apply(event_log, net, initial_marking, final_marking)
print(f"Fitness: {fitness}")

precision = precision_evaluator.apply(event_log, net, initial_marking, final_marking)
print(f"Precision: {precision}")

# 4. Convert to BPMN
bpmn_graph = pm4py.convert_to_bpmn(process_tree)
bpmn_gviz = bpmn_visualizer.apply(bpmn_graph)
bpmn_visualizer.save(bpmn_gviz, "fx_trade_bpmn.png")


# 5. Visualization
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FX Trade Process Mining Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        h1 {{ color: #333; }}
        img {{ max-width: 100%; height: auto; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>FX Trade Process Mining Results</h1>
    <h2>Petri Net</h2>
    <img src="data:image/png;base64,{image_to_base64('fx_trade_petri_net.png')}" alt="Petri Net">
    <h2>Process Tree</h2>
    <img src="data:image/png;base64,{image_to_base64('fx_trade_process_tree.png')}" alt="Process Tree">
    <h2>BPMN-like Visualization</h2>
    <img src="data:image/png;base64,{image_to_base64('fx_trade_bpmn.png')}" alt="BPMN">
</body>
</html>
"""

with open("fx_trade_process_mining_results.html", "w") as f:
    f.write(html_content)

print("Results saved to fx_trade_process_mining_results.html")