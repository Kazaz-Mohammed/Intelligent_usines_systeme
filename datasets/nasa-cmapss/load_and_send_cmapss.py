#!/usr/bin/env python
"""
Script to load NASA CMAPSS dataset and send it through the microservices pipeline
"""
import sys
import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from confluent_kafka import Producer
from typing import Optional

# Add parent directory to path to import from services
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configuration
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "sensor-data"
DATASET_DIR = Path(__file__).parent

# Sensor names (21 sensors)
SENSOR_NAMES = [
    "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr",
    "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32"
]

# Operational settings names
OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]

# Create Kafka producer
producer = Producer({
    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
    'client.id': 'nasa-cmapss-producer',
    'batch.size': 16384,
    'linger.ms': 10
})


def load_cmapss_file(filepath: Path) -> pd.DataFrame:
    """
    Load NASA CMAPSS data file
    
    Format: 26 columns space-separated
    1) unit number
    2) time (cycles)
    3-5) operational settings (3)
    6-26) sensor measurements (21)
    """
    print(f"Loading {filepath.name}...")
    
    # Column names
    columns = ['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + SENSOR_NAMES
    
    # Read file
    df = pd.read_csv(
        filepath,
        sep=' ',
        header=None,
        names=columns,
        engine='python'
    )
    
    # Remove any trailing spaces/empty columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    print(f"  Loaded {len(df)} rows, {df['unit'].nunique()} units")
    return df


def send_sensor_data(
    asset_id: str,
    sensor_id: str,
    value: float,
    cycle: int,
    timestamp: datetime,
    unit: str = "normalized",
    quality: int = 2
):
    """Send sensor data to Kafka"""
    # Format timestamp correctly: if timezone-aware, use isoformat and replace +00:00 with Z
    # If timezone-naive, assume UTC and add Z
    if timestamp.tzinfo is not None:
        # Timezone-aware: use isoformat and replace +00:00 with Z for UTC
        ts_str = timestamp.isoformat()
        if ts_str.endswith('+00:00'):
            ts_str = ts_str[:-6] + 'Z'
        elif ts_str.endswith('-00:00'):
            ts_str = ts_str[:-6] + 'Z'
    else:
        # Timezone-naive: assume UTC and add Z
        ts_str = timestamp.isoformat() + 'Z'
    
    data = {
        "timestamp": ts_str,
        "asset_id": asset_id,
        "sensor_id": sensor_id,
        "value": float(value),
        "unit": unit,
        "quality": quality,
        "source_type": "NASA_CMAPSS",
        "metadata": {
            "cycle": cycle,
            "dataset": "CMAPSS"
        }
    }
    
    try:
        producer.produce(
            KAFKA_TOPIC,
            key=f"{asset_id}:{sensor_id}:{cycle}",
            value=json.dumps(data),
            callback=lambda err, msg: None if err is None else print(f"✗ Error: {err}")
        )
        producer.poll(0)
    except Exception as e:
        print(f"Error sending {asset_id}/{sensor_id}: {e}")


def send_cmapss_data(
    df: pd.DataFrame,
    dataset_name: str = "FD001",
    max_units: Optional[int] = None,
    max_cycles_per_unit: Optional[int] = None,
    delay_ms: float = 10.0
):
    """
    Send CMAPSS data through Kafka
    
    Args:
        df: DataFrame with CMAPSS data
        dataset_name: Name of the dataset (FD001, FD002, etc.)
        max_units: Maximum number of units to process (None = all)
        max_cycles_per_unit: Maximum cycles per unit (None = all)
        delay_ms: Delay between messages in milliseconds
    """
    print(f"\nSending data from {dataset_name}...")
    print(f"  Total rows: {len(df)}")
    print(f"  Total units: {df['unit'].nunique()}")
    
    # Filter units if needed
    units = sorted(df['unit'].unique())
    if max_units:
        units = units[:max_units]
        df = df[df['unit'].isin(units)]
        print(f"  Processing {len(units)} units (limited)")
    
    # Base timestamp (start from now and go backwards)
    try:
        from datetime import timezone
        base_time = datetime.now(timezone.utc)
    except ImportError:
        base_time = datetime.utcnow()
    
    total_sent = 0
    units_processed = 0
    
    for unit_id in units:
        unit_data = df[df['unit'] == unit_id].sort_values('cycle')
        
        if max_cycles_per_unit:
            unit_data = unit_data.head(max_cycles_per_unit)
        
        # Calculate timestamps (simulate real-time, 1 cycle = 1 hour)
        cycles = unit_data['cycle'].values
        start_time = base_time - timedelta(hours=len(cycles))
        
        units_processed += 1
        unit_id_int = int(unit_id)  # Convert to int
        asset_id = f"ENGINE_{dataset_name}_{unit_id_int:03d}"
        
        print(f"  Processing unit {unit_id_int} ({units_processed}/{len(units)}): {len(unit_data)} cycles", end="", flush=True)
        
        # Send operational settings as sensors
        for cycle_idx, (idx, row) in enumerate(unit_data.iterrows()):
            cycle = int(row['cycle'])
            timestamp = start_time + timedelta(hours=cycle_idx)
            
            # Send operational settings
            for i, op_name in enumerate(OP_SETTINGS, 1):
                send_sensor_data(
                    asset_id=asset_id,
                    sensor_id=f"OP_SETTING_{i}",
                    value=float(row[op_name]),
                    cycle=cycle,
                    timestamp=timestamp,
                    unit="normalized",
                    quality=2
                )
                total_sent += 1
            
            # Send sensor measurements
            for sensor_name in SENSOR_NAMES:
                if sensor_name in row and pd.notna(row[sensor_name]):
                    send_sensor_data(
                        asset_id=asset_id,
                        sensor_id=sensor_name,
                        value=float(row[sensor_name]),
                        cycle=cycle,
                        timestamp=timestamp,
                        unit="normalized",
                        quality=2
                    )
                    total_sent += 1
            
            # Small delay to avoid overwhelming Kafka
            if delay_ms > 0 and cycle_idx % 10 == 0:
                time.sleep(delay_ms / 1000.0)
        
        print(f" [OK] ({total_sent} messages sent)")
        
        # Flush periodically
        if units_processed % 5 == 0:
            producer.flush()
    
    # Final flush
    producer.flush(10)
    print(f"\n[SUCCESS] Total messages sent: {total_sent}")
    print(f"[SUCCESS] Processed {units_processed} units from {dataset_name}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and send NASA CMAPSS data to Kafka")
    parser.add_argument(
        "-f", "--file",
        type=str,
        default="train_FD001.txt",
        help="CMAPSS file to load (default: train_FD001.txt)"
    )
    parser.add_argument(
        "-u", "--max-units",
        type=int,
        default=5,
        help="Maximum number of units to process (default: 5, use 0 for all)"
    )
    parser.add_argument(
        "-c", "--max-cycles",
        type=int,
        default=100,
        help="Maximum cycles per unit (default: 100, use 0 for all)"
    )
    parser.add_argument(
        "-d", "--delay",
        type=float,
        default=10.0,
        help="Delay between message batches in ms (default: 10.0)"
    )
    
    args = parser.parse_args()
    
    # File path
    filepath = DATASET_DIR / args.file
    if not filepath.exists():
        print(f"❌ Error: File not found: {filepath}")
        sys.exit(1)
    
    # Load data
    df = load_cmapss_file(filepath)
    
    # Extract dataset name from filename
    dataset_name = args.file.replace("train_", "").replace("test_", "").replace(".txt", "")
    
    # Send data
    max_units = args.max_units if args.max_units > 0 else None
    max_cycles = args.max_cycles if args.max_cycles > 0 else None
    
    print(f"\n{'='*60}")
    print(f"Sending to Kafka topic: {KAFKA_TOPIC}")
    print(f"Configuration:")
    print(f"  - Max units: {max_units or 'all'}")
    print(f"  - Max cycles per unit: {max_cycles or 'all'}")
    print(f"  - Delay: {args.delay}ms")
    print(f"{'='*60}\n")
    
    send_cmapss_data(
        df=df,
        dataset_name=dataset_name,
        max_units=max_units,
        max_cycles_per_unit=max_cycles,
        delay_ms=args.delay
    )
    
    print(f"\n[SUCCESS] Done! Check your microservices logs to see data being processed.")


if __name__ == "__main__":
    main()

