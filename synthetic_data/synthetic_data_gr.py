import random
from datetime import datetime, timedelta
import pandas as pd
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FXOptionsConfig:
    traders: List[str] = None
    currencies: List[str] = None
    option_types: List[str] = None
    trading_strategies: List[str] = None

    def __post_init__(self):
        self.traders = ["Options Desk A", "Market Maker B", "Hedge Desk C", "Client Desk D"]
        self.currencies = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "EUR/GBP", "USD/CHF"]
        self.option_types = ["Vanilla Call", "Vanilla Put", "Butterfly", "Risk Reversal", "Straddle", "Strangle"]
        self.trading_strategies = ["Delta Hedge", "Gamma Scalping", "Vega Trading", "Vol Surface Trading"]
        self.client_types = ["Hedge Fund", "Corporate", "Bank", "Asset Manager"]
        self.booking_systems = ["Murex", "Summit", "Calypso", "Internal"]


def generate_fx_trade_event_log(num_cases=3000):
    """Generate synthetic FX options trade event log"""
    config = FXOptionsConfig()
    event_types = [
        "Trade Initiated",
        "Market Data Validation",
        "Volatility Surface Analysis",
        "Premium Calculation",
        "Quote Requested",
        "Quote Provided",
        "Trade Execution",
        "Trade Validation",
        "Initial Margin Calculation",
        "Credit Check",
        "Risk Assessment",
        "Strategy Validation",
        "Greeks Calculation",
        "Trade Allocation",
        "Trade Confirmation",
        "Trade Matching",
        "Settlement Instructions",
        "Collateral Management",
        "Premium Settlement",
        "Position Reconciliation",
        "Exercise Decision",
        "Final Settlement",
        "Trade Reconciliation"
    ]

    # Additional activity sets
    regulatory_activities = [
        "Transaction Reporting Check",
        "Best Execution Validation",
        "Trade Transparency Assessment",
        "Regulatory Reporting Generation"
    ]

    market_making_activities = [
        "Volatility Smile Analysis",
        "Skew Calibration",
        "Surface Construction",
        "Market Making Spread Calculation"
    ]

    client_documentation_activities = [
        "ISDA Master Agreement Check",
        "CSA Verification",
        "Client Limit Validation",
        "KYC Refresh Check"
    ]

    event_log = []

    for case_id in range(1, num_cases + 1):
        # Determine trade characteristics
        client_type = random.choice(config.client_types)
        option_type = random.choice(config.option_types)
        trading_strategy = random.choice(config.trading_strategies)
        currency = random.choice(config.currencies)
        trader = random.choice(config.traders)
        booking_system = random.choice(config.booking_systems)

        # Build activity sequence
        case_events = event_types.copy()

        # Add conditional activities based on trade characteristics
        if client_type in ["Hedge Fund", "Asset Manager"]:
            case_events.extend(regulatory_activities)

        if "Market Maker" in trader:
            case_events.extend(market_making_activities)

        if random.random() < 0.4:  # 40% chance of documentation review
            case_events.extend(client_documentation_activities)

        # Randomize order of some activities while maintaining core sequence
        core_sequence = case_events[:7]  # Keep initial sequence fixed
        middle_sequence = case_events[7:-5]  # Randomize middle activities
        end_sequence = case_events[-5:]  # Keep end sequence fixed

        random.shuffle(middle_sequence)
        case_events = core_sequence + middle_sequence + end_sequence

        # Generate events with timestamps
        start_time = datetime.now() + timedelta(days=random.randint(-30, 0))

        for i, event in enumerate(case_events):
            timestamp = start_time + timedelta(minutes=random.randint(10, 60) * i)

            # Base event data
            event_data = {
                "case_id": f"Case_{case_id}",
                "activity": event,
                "timestamp": timestamp,
                "resource": trader,
                "currency_pair": currency,
                "option_type": option_type,
                "booking_system": booking_system,
                "client_type": client_type,
                "trading_strategy": trading_strategy
            }

            # Add activity-specific fields
            if event == "Trade Execution":
                event_data.update({
                    "strike_price": round(random.uniform(0.9, 1.1), 4),
                    "premium": round(random.uniform(0.001, 0.05), 4),
                    "notional_amount": random.randint(1000000, 50000000),
                })
            elif "Risk Assessment" in event:
                event_data.update({
                    "risk_score": round(random.uniform(1, 5), 2),
                    "limit_usage": f"{random.randint(0, 100)}%"
                })
            elif "Greeks" in event:
                event_data.update({
                    "delta": round(random.uniform(-1, 1), 2),
                    "gamma": round(random.uniform(0, 0.2), 3),
                    "vega": round(random.uniform(0, 50000), 2),
                    "theta": round(random.uniform(-1000, 0), 2)
                })

            event_log.append(event_data)

    df = pd.DataFrame(event_log)
    df.sort_values(by=["case_id", "timestamp"], inplace=True)
    return df


def main():
    # Generate the event log
    fx_event_log = generate_fx_trade_event_log(3000)

    # Save CSV with semicolon separator
    output_path = "../staging/fx_trade_log.csv"
    fx_event_log.to_csv(output_path, sep=';', index=False)

    logger.info(f"Generated {len(fx_event_log)} activities")
    logger.info(f"Event log saved to: {output_path}")


if __name__ == "__main__":
    main()