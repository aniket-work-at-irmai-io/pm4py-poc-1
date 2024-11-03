import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer

# Load the XES file
event_log = pm4py.read_xes("fx_trade_log.xes")

# Apply Inductive Miner
process_tree = inductive_miner.apply_tree(event_log)

# Convert to Petri Net
net, initial_marking, final_marking = pm4py.convert_to_petri_net(process_tree)

# Visualize Petri Net
pn_gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.save(pn_gviz, "output/fx_trade_petri_net.png")

# Visualize Process Tree
pt_gviz = pt_visualizer.apply(process_tree)
pt_visualizer.save(pt_gviz, "output/fx_trade_process_tree.png")