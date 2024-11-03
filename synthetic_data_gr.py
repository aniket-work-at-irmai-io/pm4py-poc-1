import pandas as pd
import random
from datetime import datetime, timedelta
import pm4py


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

# Convert to XES format
event_log_xes = pm4py.convert_to_event_log(fx_event_log)
pm4py.write_xes(event_log_xes, "fx_trade_log.xes")