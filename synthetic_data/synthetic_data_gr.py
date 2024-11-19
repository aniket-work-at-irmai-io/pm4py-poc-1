import random
from datetime import datetime, timedelta
import pandas as pd
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import json
from collections import Counter

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fx_trade_generation.log'),
        logging.StreamHandler()
    ]
)
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


class MetadataTracker:
    """Tracks metadata about generated event log"""

    def __init__(self):
        self.total_events = 0
        self.total_cases = 0
        self.activity_counts = Counter()
        self.field_usage = Counter()
        self.client_type_distribution = Counter()
        self.option_type_distribution = Counter()
        self.currency_pair_distribution = Counter()
        self.avg_activities_per_case = 0
        self.path_variations = set()

    def to_dict(self):
        return {
            "total_events": self.total_events,
            "total_cases": self.total_cases,
            "activity_distribution": dict(self.activity_counts),
            "field_usage": dict(self.field_usage),
            "client_type_distribution": dict(self.client_type_distribution),
            "option_type_distribution": dict(self.option_type_distribution),
            "currency_pair_distribution": dict(self.currency_pair_distribution),
            "avg_activities_per_case": self.avg_activities_per_case,
            "unique_path_variations": len(self.path_variations)
        }


def generate_fx_trade_event_log(num_cases=30000):
    """Generate synthetic FX options trade event log with detailed tracking"""
    config = FXOptionsConfig()
    metadata = MetadataTracker()

    logger.info(f"Starting event log generation for {num_cases} cases")

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

    logger.info(f"Core activities: {len(event_types)}")
    logger.info(f"Regulatory activities: {len(regulatory_activities)}")
    logger.info(f"Market making activities: {len(market_making_activities)}")
    logger.info(f"Documentation activities: {len(client_documentation_activities)}")

    event_log = []

    for case_id in range(1, num_cases + 1):
        if case_id % 1000 == 0:
            logger.info(f"Generated {case_id} cases...")

        # Determine trade characteristics
        client_type = random.choice(config.client_types)
        option_type = random.choice(config.option_types)
        trading_strategy = random.choice(config.trading_strategies)
        currency = random.choice(config.currencies)
        trader = random.choice(config.traders)
        booking_system = random.choice(config.booking_systems)

        # Track distributions
        metadata.client_type_distribution[client_type] += 1
        metadata.option_type_distribution[option_type] += 1
        metadata.currency_pair_distribution[currency] += 1

        # Build activity sequence
        case_events = event_types.copy()

        # Add conditional activities based on trade characteristics
        if client_type in ["Hedge Fund", "Asset Manager"]:
            case_events.extend(regulatory_activities)

        if "Market Maker" in trader:
            case_events.extend(market_making_activities)

        if random.random() < 0.4:  # 40% chance of documentation review
            case_events.extend(client_documentation_activities)

        # Track path variation
        metadata.path_variations.add(tuple(case_events))

        # Randomize order of some activities while maintaining core sequence
        core_sequence = case_events[:7]  # Keep initial sequence fixed
        middle_sequence = case_events[7:-5]  # Randomize middle activities
        end_sequence = case_events[-5:]  # Keep end sequence fixed

        random.shuffle(middle_sequence)
        case_events = core_sequence + middle_sequence + end_sequence

        # Generate events with timestamps
        start_time = datetime.now() + timedelta(days=random.randint(-30, 0))

        case_activity_count = 0

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

            # Track metadata
            metadata.activity_counts[event] += 1
            metadata.field_usage.update(event_data.keys())
            case_activity_count += 1

            event_log.append(event_data)

        metadata.total_events += case_activity_count

    metadata.total_cases = num_cases
    metadata.avg_activities_per_case = metadata.total_events / num_cases

    # Log metadata
    logger.info("\n=== Event Log Generation Complete ===")
    logger.info(f"Total events generated: {metadata.total_events}")
    logger.info(f"Average activities per case: {metadata.avg_activities_per_case:.2f}")
    logger.info(f"Unique path variations: {len(metadata.path_variations)}")
    logger.info("\nActivity Distribution:")
    for activity, count in metadata.activity_counts.most_common(10):
        logger.info(f"  {activity}: {count}")
    logger.info("\nField Usage:")
    for field, count in metadata.field_usage.most_common():
        logger.info(f"  {field}: {count}")

    # Save metadata to JSON
    with open('event_log_metadata.json', 'w') as f:
        json.dump(metadata.to_dict(), f, indent=4)

    df = pd.DataFrame(event_log)
    df.sort_values(by=["case_id", "timestamp"], inplace=True)

    # Log DataFrame statistics
    logger.info("\n=== DataFrame Statistics ===")
    logger.info(f"Shape: {df.shape}")
    logger.info("\nColumn Info:")
    for col in df.columns:
        logger.info(f"  {col}: {df[col].nunique()} unique values")

    return df


def main():
    start_time = datetime.now()
    logger.info(f"Starting FX trade event log generation at {start_time}")

    # Generate the event log
    fx_event_log = generate_fx_trade_event_log(30000)

    # Save CSV with semicolon separator
    output_path = "fx_trade_log.csv"
    fx_event_log.to_csv(output_path, sep=';', index=False)

    end_time = datetime.now()
    duration = end_time - start_time

    logger.info(f"\n=== Generation Complete ===")
    logger.info(f"Duration: {duration}")
    logger.info(f"Output saved to: {output_path}")
    logger.info(f"Metadata saved to: event_log_metadata.json")


if __name__ == "__main__":
    main()