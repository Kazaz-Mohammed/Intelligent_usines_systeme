"""
Script de test de charge pour envoyer un grand nombre de données
"""
import json
import time
import argparse
from datetime import datetime
from confluent_kafka import Producer

# Configuration Kafka
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "sensor-data"

# Créer le producer
producer = Producer({
    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
    'client.id': 'load-test-producer',
    'batch.size': 16384,
    'linger.ms': 10
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
    
    try:
        producer.produce(
            KAFKA_TOPIC,
            key=f"{asset_id}:{sensor_id}",
            value=json.dumps(data)
        )
    except Exception as e:
        print(f"Erreur lors de l'envoi: {e}")


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Test de charge pour données capteurs")
    parser.add_argument("--count", type=int, default=1000, help="Nombre de messages à envoyer")
    parser.add_argument("--interval", type=float, default=0.01, help="Intervalle entre les messages (secondes)")
    
    args = parser.parse_args()
    
    print(f"Envoi de {args.count} messages vers {KAFKA_TOPIC}...")
    
    start_time = time.time()
    
    for i in range(args.count):
        # Générer des données variées
        asset_id = f"ASSET{(i % 3) + 1:03d}"
        sensor_id = f"SENSOR{(i % 5) + 1:03d}"
        value = 25.0 + (i % 100) * 0.1
        
        send_sensor_data(asset_id, sensor_id, value, "°C")
        
        if (i + 1) % 100 == 0:
            print(f"Envoyé {i + 1}/{args.count} messages...")
            producer.poll(0)
        
        time.sleep(args.interval)
    
    # Attendre que tous les messages soient envoyés
    producer.flush()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Test terminé!")
    print(f"Messages envoyés: {args.count}")
    print(f"Temps écoulé: {elapsed_time:.2f} secondes")
    print(f"Débit: {args.count / elapsed_time:.2f} messages/seconde")


if __name__ == "__main__":
    main()

