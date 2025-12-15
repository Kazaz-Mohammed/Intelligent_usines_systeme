#!/usr/bin/env python
"""Quick test script to send sensor data to Kafka"""
import json
import time
from datetime import datetime
from confluent_kafka import Producer

# Configuration
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "sensor-data"

# Create producer
producer = Producer({
    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
    'client.id': 'test-producer'
})

def send_sensor_data(asset_id: str, sensor_id: str, value: float, unit: str = "°C"):
    """Send sensor data"""
    data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "asset_id": asset_id,
        "sensor_id": sensor_id,
        "value": value,
        "unit": unit,
        "quality": 2,  # 2 = good quality
        "source_type": "TEST",
        "metadata": {}
    }
    
    try:
        producer.produce(
            KAFKA_TOPIC,
            key=f"{asset_id}:{sensor_id}",
            value=json.dumps(data),
            callback=lambda err, msg: print(f"✓ Sent: {msg.key()}") if err is None else print(f"✗ Error: {err}")
        )
        producer.poll(0)
        print(f"Sent: {asset_id}/{sensor_id} = {value} {unit}")
    except Exception as e:
        print(f"Error sending: {e}")

if __name__ == "__main__":
    print(f"Sending test data to topic: {KAFKA_TOPIC}")
    print("=" * 60)
    
    # Send a few test messages
    for i in range(5):
        send_sensor_data("ASSET001", "SENSOR001", 25.0 + i * 0.1, "°C")
        send_sensor_data("ASSET001", "SENSOR002", 100.0 + i * 0.5, "bar")
        time.sleep(0.2)
    
    # Flush to ensure all messages are sent
    producer.flush(10)
    print("=" * 60)
    print("✓ All messages sent!")

