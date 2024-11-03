import pandas as pd
import pm4py

df = pd.read_csv('staging/running_example.csv', sep=';')
print(df)


log = pm4py.format_dataframe(df, case_id='case_id',activity_key='activity',
                             timestamp_key='timestamp')

print(log)

print(pm4py.get_start_activities(log))
print(pm4py.get_end_activities(log))


# final process map
log.to_csv('output/running_example_exported.csv')
