from pm4py.algo.analysis.woflan import algorithm as woflan
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

# Soundness check
is_sound = woflan.apply(net, initial_marking, final_marking)
print(f"Is the Petri Net sound? {is_sound}")

# Fitness evaluation
fitness = replay_fitness.apply(event_log, net, initial_marking, final_marking)
print(f"Fitness: {fitness}")

# Precision evaluation
precision = precision_evaluator.apply(event_log, net, initial_marking, final_marking)
print(f"Precision: {precision}")