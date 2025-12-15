#!/usr/bin/env python
"""
Script pour entraîner les modèles de détection d'anomalies et de prédiction RUL
en utilisant les features extraites depuis Kafka ou la base de données.
"""
import sys
import os
import json
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime, timezone
import logging

# Add parent directory to path to import from services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DETECTION_ANOMALIES_URL = "http://localhost:8084"
PREDICTION_RUL_URL = "http://localhost:8085"
EXTRACTION_FEATURES_URL = "http://localhost:8083"


def collect_features_from_kafka(asset_id: str, max_messages: int = 1000) -> List[Dict[str, Any]]:
    """
    Collecte les features depuis Kafka en consommant le topic extracted-features.
    
    Args:
        asset_id: ID de l'actif à filtrer (ou None pour tous)
        max_messages: Nombre maximum de messages à consommer
    
    Returns:
        Liste de messages de features
    """
    try:
        from confluent_kafka import Consumer, KafkaError
        
        config = {
            'bootstrap.servers': 'localhost:9092',
            'group.id': 'training-script',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False
        }
        
        consumer = Consumer(config)
        consumer.subscribe(['extracted-features'])
        
        messages = []
        logger.info(f"Collecte de {max_messages} messages depuis Kafka...")
        
        while len(messages) < max_messages:
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                if len(messages) == 0:
                    continue  # Continue waiting
                else:
                    break  # No more messages
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    break
                logger.error(f"Erreur Kafka: {msg.error()}")
                continue
            
            try:
                data = json.loads(msg.value().decode('utf-8'))
                if asset_id is None or data.get('asset_id') == asset_id:
                    messages.append(data)
                    if len(messages) % 100 == 0:
                        logger.info(f"  Collecté {len(messages)} messages...")
            except Exception as e:
                logger.warning(f"Erreur lors de la désérialisation: {e}")
                continue
        
        consumer.close()
        logger.info(f"✓ {len(messages)} messages collectés depuis Kafka")
        return messages
        
    except ImportError:
        logger.warning("confluent_kafka non disponible, utilisation de l'API REST")
        return []
    except Exception as e:
        logger.error(f"Erreur lors de la collecte depuis Kafka: {e}")
        return []


def collect_features_from_api(asset_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Collecte les features depuis l'API REST du service extraction-features.
    
    Args:
        asset_id: ID de l'actif
        limit: Nombre maximum de features à récupérer
    
    Returns:
        Liste de features
    """
    try:
        # Try the correct endpoint: /api/v1/features/features/{asset_id}
        url = f"{EXTRACTION_FEATURES_URL}/api/v1/features/features/{asset_id}?limit={limit}"
        logger.info(f"Collecte des features depuis: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        features = data.get('features', [])
        count = data.get('count', len(features))
        
        logger.info(f"✓ {count} features collectées depuis l'API (asset_id: {asset_id})")
        
        if count == 0:
            logger.warning(f"Aucune feature trouvée pour {asset_id}. Vérifiez que:")
            logger.warning("  1. Le service extraction-features est en cours d'exécution")
            logger.warning("  2. Des données ont été envoyées et traitées")
            logger.warning("  3. L'asset_id est correct")
            logger.warning("  Essayez --source kafka ou --skip-collection pour utiliser des données de test")
        
        return features
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de la collecte depuis l'API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                logger.error(f"  Détails: {error_detail}")
            except:
                logger.error(f"  Réponse: {e.response.text}")
        return []
    except Exception as e:
        logger.error(f"Erreur lors de la collecte depuis l'API: {e}", exc_info=True)
        return []


def group_features_by_timestamp(feature_messages: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    Groupe les features par timestamp pour créer des vecteurs de features.
    
    Args:
        feature_messages: Liste de messages de features (format Kafka ou API)
    
    Returns:
        Liste de dictionnaires {feature_name: feature_value} par timestamp
    """
    # Grouper par (asset_id, sensor_id, timestamp)
    grouped = defaultdict(lambda: defaultdict(dict))
    
    for msg in feature_messages:
        # Support both formats: Kafka message format or API format
        if 'features' in msg:
            # Kafka format: {"asset_id": "...", "sensor_id": "...", "features": {...}, "timestamp": "..."}
            asset_id = msg.get('asset_id')
            sensor_id = msg.get('sensor_id', 'unknown')
            timestamp = msg.get('timestamp')
            features = msg.get('features', {})
            
            if asset_id and timestamp:
                key = (asset_id, sensor_id, timestamp)
                grouped[key].update(features)
        else:
            # API format: {"name": "...", "value": ..., "metadata": {"asset_id": "...", "timestamp": "..."}}
            name = msg.get('name')
            value = msg.get('value')
            metadata = msg.get('metadata', {})
            
            asset_id = metadata.get('asset_id')
            sensor_id = metadata.get('sensor_id', 'unknown')
            timestamp = metadata.get('timestamp')
            
            if asset_id and timestamp and name is not None and value is not None:
                key = (asset_id, sensor_id, timestamp)
                grouped[key][name] = value
    
    # Convertir en liste de vecteurs de features
    feature_vectors = []
    for (asset_id, sensor_id, timestamp), features in grouped.items():
        if features:  # Only add if we have features
            feature_vectors.append({
                'asset_id': asset_id,
                'sensor_id': sensor_id,
                'timestamp': timestamp,
                'features': features
            })
    
    logger.info(f"✓ {len(feature_vectors)} vecteurs de features créés")
    return feature_vectors


def prepare_anomaly_training_data(feature_vectors: List[Dict[str, Any]]) -> tuple:
    """
    Prépare les données d'entraînement pour la détection d'anomalies.
    
    Args:
        feature_vectors: Liste de vecteurs de features
    
    Returns:
        (data_array, feature_names) où data_array est un array 2D numpy
    """
    if not feature_vectors:
        raise ValueError("Aucun vecteur de features disponible")
    
    # Collecter tous les noms de features uniques
    all_feature_names = set()
    for vec in feature_vectors:
        all_feature_names.update(vec['features'].keys())
    
    feature_names = sorted(list(all_feature_names))
    logger.info(f"  {len(feature_names)} features uniques trouvées")
    
    # Créer un array 2D
    data = []
    for vec in feature_vectors:
        row = [vec['features'].get(name, 0.0) for name in feature_names]
        data.append(row)
    
    data_array = np.array(data, dtype=np.float64)
    logger.info(f"✓ Données d'entraînement préparées: shape={data_array.shape}")
    
    return data_array, feature_names


def prepare_rul_training_data(feature_vectors: List[Dict[str, Any]]) -> tuple:
    """
    Prépare les données d'entraînement pour la prédiction RUL.
    Calcule le RUL à partir des cycles dans les métadonnées.
    
    Args:
        feature_vectors: Liste de vecteurs de features
    
    Returns:
        (training_data, target_data, feature_names)
    """
    if not feature_vectors:
        raise ValueError("Aucun vecteur de features disponible")
    
    # Grouper par asset_id pour calculer le RUL
    asset_data = defaultdict(list)
    for vec in feature_vectors:
        asset_id = vec['asset_id']
        asset_data[asset_id].append(vec)
    
    # Pour chaque asset, calculer le RUL (max_cycle - current_cycle)
    all_training_data = []
    all_target_data = []
    all_feature_names = set()
    
    for asset_id, vectors in asset_data.items():
        # Trier par timestamp
        vectors.sort(key=lambda v: v.get('timestamp', ''))
        
        # Trouver le cycle maximum dans les métadonnées
        max_cycle = 0
        for vec in vectors:
            # Chercher cycle dans les features ou métadonnées
            metadata = vec.get('metadata', {})
            cycle = metadata.get('cycle')
            if cycle is None:
                # Essayer de trouver dans les features
                cycle = vec['features'].get('cycle')
            if cycle is not None:
                max_cycle = max(max_cycle, int(cycle))
        
        # Si pas de cycle trouvé, utiliser le nombre de vecteurs comme proxy
        if max_cycle == 0:
            max_cycle = len(vectors)
            logger.warning(f"  Cycle maximum non trouvé pour {asset_id}, utilisation de {max_cycle}")
        
        # Créer les données d'entraînement
        for i, vec in enumerate(vectors):
            # Calculer le RUL (cycles restants)
            current_cycle = metadata.get('cycle') if 'metadata' in vec else None
            if current_cycle is None:
                current_cycle = i + 1  # Approximate
            
            rul = max(0, max_cycle - int(current_cycle))
            
            # Ajouter les features
            all_feature_names.update(vec['features'].keys())
            all_training_data.append(vec['features'])
            all_target_data.append(rul)
    
    feature_names = sorted(list(all_feature_names))
    logger.info(f"  {len(feature_names)} features uniques trouvées")
    
    # Convertir en arrays numpy
    training_data = []
    for features in all_training_data:
        row = [features.get(name, 0.0) for name in feature_names]
        training_data.append(row)
    
    training_array = np.array(training_data, dtype=np.float64)
    target_array = np.array(all_target_data, dtype=np.float64)
    
    logger.info(f"✓ Données d'entraînement RUL préparées: shape={training_array.shape}, targets={len(target_array)}")
    
    return training_array, target_array, feature_names


def train_anomaly_detection_models(data: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    """
    Entraîne les modèles de détection d'anomalies.
    
    Args:
        data: Array 2D numpy avec les features
        feature_names: Liste des noms de features
    
    Returns:
        Résultats de l'entraînement
    """
    logger.info("=" * 60)
    logger.info("Entraînement des modèles de détection d'anomalies")
    logger.info("=" * 60)
    
    url = f"{DETECTION_ANOMALIES_URL}/api/v1/anomalies/train"
    
    payload = {
        "data": data.tolist(),
        "feature_names": feature_names,
        "model_names": ["isolation_forest", "one_class_svm"]  # Exclure LSTM pour l'instant
    }
    
    try:
        logger.info(f"Envoi de la requête à {url}...")
        response = requests.post(url, json=payload, timeout=600)  # 10 minutes timeout
        response.raise_for_status()
        
        results = response.json()
        logger.info("✓ Modèles entraînés avec succès!")
        
        for model_name, result in results.items():
            status = result.get('status', 'unknown')
            message = result.get('message', '')
            logger.info(f"  {model_name}: {status} - {message}")
        
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de l'entraînement: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"  Réponse: {e.response.text}")
        raise


def train_rul_prediction_models(training_data: np.ndarray, target_data: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    """
    Entraîne les modèles de prédiction RUL.
    
    Args:
        training_data: Array 2D numpy avec les features
        target_data: Array 1D numpy avec les valeurs RUL
        feature_names: Liste des noms de features
    
    Returns:
        Résultats de l'entraînement
    """
    logger.info("=" * 60)
    logger.info("Entraînement des modèles de prédiction RUL")
    logger.info("=" * 60)
    
    url = f"{PREDICTION_RUL_URL}/api/v1/rul/train"
    
    payload = {
        "training_data": training_data.tolist(),
        "target_data": target_data.tolist(),
        "feature_names": feature_names,
        "model_name": None,  # Entraîner tous les modèles
        "parameters": {
            "epochs": 50,  # Pour les modèles deep learning (si activés)
            "batch_size": 32
        }
    }
    
    try:
        logger.info(f"Envoi de la requête à {url}...")
        response = requests.post(url, json=payload, timeout=600)  # 10 minutes timeout
        response.raise_for_status()
        
        results = response.json()
        logger.info("✓ Modèles entraînés avec succès!")
        
        for model_name, result in results.items():
            status = result.get('status', 'unknown')
            message = result.get('message', '')
            logger.info(f"  {model_name}: {status} - {message}")
        
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de l'entraînement: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"  Réponse: {e.response.text}")
        raise


def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraîner les modèles ML")
    parser.add_argument("--asset-id", type=str, default="ENGINE_FD001_000", help="ID de l'actif (défaut: ENGINE_FD001_000)")
    parser.add_argument("--source", choices=["kafka", "api"], default="api", help="Source des données (défaut: api)")
    parser.add_argument("--max-messages", type=int, default=1000, help="Nombre maximum de messages à collecter")
    parser.add_argument("--train-anomaly", action="store_true", default=True, help="Entraîner les modèles de détection d'anomalies")
    parser.add_argument("--train-rul", action="store_true", default=True, help="Entraîner les modèles de prédiction RUL")
    parser.add_argument("--skip-collection", action="store_true", help="Sauter la collecte (utiliser des données de test)")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Script d'entraînement des modèles ML")
    logger.info("=" * 60)
    
    # Collecter les features
    if not args.skip_collection:
        if args.source == "kafka":
            feature_messages = collect_features_from_kafka(args.asset_id, args.max_messages)
        else:
            feature_messages = collect_features_from_api(args.asset_id, args.max_messages)
        
        if not feature_messages:
            logger.error("Aucune feature collectée. Vérifiez que les services sont en cours d'exécution et qu'il y a des données.")
            logger.error("")
            logger.error("Solutions possibles:")
            logger.error("  1. Utilisez --source kafka pour collecter depuis Kafka")
            logger.error("  2. Utilisez --skip-collection pour utiliser des données de test")
            logger.error("  3. Vérifiez que des données ont été envoyées et traitées")
            logger.error("  4. Vérifiez que l'asset_id est correct (actuellement: {})".format(args.asset_id))
            return
        
        # Grouper les features
        feature_vectors = group_features_by_timestamp(feature_messages)
        
        if not feature_vectors:
            logger.error("Aucun vecteur de features créé.")
            return
    else:
        # Utiliser des données de test
        logger.info("Utilisation de données de test...")
        feature_vectors = [
            {
                'asset_id': args.asset_id,
                'sensor_id': 'sensor1',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'features': {f'feature_{i}': float(i) for i in range(10)},
                'metadata': {'cycle': i}
            }
            for i in range(100)
        ]
    
    # Entraîner les modèles de détection d'anomalies
    if args.train_anomaly:
        try:
            data_array, feature_names = prepare_anomaly_training_data(feature_vectors)
            train_anomaly_detection_models(data_array, feature_names)
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement des modèles d'anomalies: {e}", exc_info=True)
    
    # Entraîner les modèles de prédiction RUL
    if args.train_rul:
        try:
            training_data, target_data, feature_names = prepare_rul_training_data(feature_vectors)
            train_rul_prediction_models(training_data, target_data, feature_names)
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement des modèles RUL: {e}", exc_info=True)
    
    logger.info("=" * 60)
    logger.info("✓ Entraînement terminé!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

