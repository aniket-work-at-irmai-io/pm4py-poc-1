bpmn_graph = pm4py.convert_to_bpmn(process_tree)
bpmn_gviz = bpmn_visualizer.apply(bpmn_graph)
bpmn_visualizer.save(bpmn_gviz, "fx_trade_bpmn.png")