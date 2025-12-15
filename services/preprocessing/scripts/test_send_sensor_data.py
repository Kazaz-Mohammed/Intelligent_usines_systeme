"""
Script de test pour envoyer des données capteurs vers Kafka
"""
import json
import time
from datetime import datetime
from confluent_kafka import Producer

# Configuration Kafka
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "sensor-data"

# Créer le producer
producer = Producer({
    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
    'client.id': 'test-sensor-data-producer'
})


def send_sensor_data(asset_id: str, sensor_id: str, value: float, unit: str = "°C"):
    """Envoyer une donnée capteur"""
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "asset_id": asset_id,
        "sensor_id": sensor_id,
        "value": value,
        "unit": unit,
        "quality": 2,
        "source_type": "TEST"
    }
    
    # Envoyer le message
    producer.produce(
        KAFKA_TOPIC,
        key=f"{asset_id}:{sensor_id}",
        value=json.dumps(data),
        callback=lambda err, msg: print(f"Message envoyé: {msg.key()}") if err is None else print(f"Erreur: {err}")
    )
    
    producer.poll(0)


def main():
    """Fonction principale"""
    print(f"Envoi de données de test vers {KAFKA_TOPIC}...")
    
    # Envoyer quelques données de test
    for i in range(10):
        # Température
        send_sensor_data("ASSET001", "SENSOR001", 25.0 + i * 0.1, "°C")
        
        # Pression
        send_sensor_data("ASSET001", "SENSOR002", 100.0 + i * 0.5, "bar")
        
        # Vibration
        send_sensor_data("ASSET001", "SENSOR003", 10.0 + i * 0.2, "mm/s")
        
        time.sleep(0.5)
    
    # Attendre que tous les messages soient envoyés
    producer.flush()
    
    print("Données envoyées avec succès!")


if __name__ == "__main__":
    main()

