# Guide Worker Kafka - Service Prediction RUL

## Vue d'ensemble

Le worker Kafka consomme les features extraites depuis Kafka, prédit la RUL en temps-réel, et publie les prédictions sur Kafka.

## Architecture

```
Kafka Topic: extracted-features
    ↓
Kafka Consumer
    ↓
RUL Prediction Service
    ↓
Kafka Producer
    ↓
Kafka Topic: rul-predictions
```

## Fonctionnalités

### 1. Consommation Kafka ✅
- Consomme les features depuis `extracted-features`
- Gestion des erreurs et messages invalides
- Support pour consommation continue ou batch

### 2. Prédiction RUL ✅
- Prédiction en temps-réel pour chaque feature
- Utilisation de l'ensemble de modèles
- Intervalles de confiance calculés

### 3. Publication Kafka ✅
- Publie les prédictions sur `rul-predictions`
- Format JSON avec toutes les métadonnées
- Key = asset_id pour partitionnement

## Utilisation

### Démarrer le Worker

```bash
# Depuis le répertoire du service
python -m app.worker
```

Ou avec uvicorn (si configuré) :

```bash
uvicorn app.worker:main
```

### Configuration

Dans `app/config.py` ou `.env` :

```python
# Kafka
kafka_bootstrap_servers: str = "localhost:9092"
kafka_consumer_group: str = "prediction-rul-service"
kafka_topic_input_features: str = "extracted-features"
kafka_topic_output_rul: str = "rul-predictions"
kafka_auto_offset_reset: str = "earliest"
kafka_enable_auto_commit: bool = True
```

### Prérequis

1. **Kafka doit être démarré**
   ```bash
   # Vérifier que Kafka est accessible
   kafka-topics --list --bootstrap-server localhost:9092
   ```

2. **Les modèles doivent être entraînés**
   ```bash
   # Entraîner les modèles via l'API
   curl -X POST "http://localhost:8085/api/v1/rul/train" \
     -H "Content-Type: application/json" \
     -d '{
       "training_data": [...],
       "target_data": [...]
     }'
   ```

3. **Le topic `extracted-features` doit exister**
   ```bash
   # Créer le topic si nécessaire
   kafka-topics --create \
     --bootstrap-server localhost:9092 \
     --topic extracted-features \
     --partitions 3 \
     --replication-factor 1
   ```

## Format des Messages

### Input : Features Extraites

```json
{
  "asset_id": "ASSET001",
  "sensor_id": "SENSOR001",
  "features": {
    "rms": 10.5,
    "kurtosis": 2.3,
    "crest_factor": 3.1
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "metadata": {
    "source": "extraction-features-service"
  }
}
```

### Output : Prédiction RUL

```json
{
  "asset_id": "ASSET001",
  "sensor_id": "SENSOR001",
  "timestamp": "2024-01-01T12:00:00Z",
  "rul_prediction": 150.5,
  "confidence_interval_lower": 140.0,
  "confidence_interval_upper": 160.0,
  "confidence_level": 0.95,
  "uncertainty": 10.0,
  "model_used": "ensemble",
  "model_scores": {
    "lstm": 150.0,
    "gru": 151.0,
    "tcn": 150.5
  },
  "features": {
    "rms": 10.5,
    "kurtosis": 2.3
  },
  "metadata": {}
}
```

## Flux de Traitement

1. **Réception** : Le worker reçoit un message de features depuis Kafka
2. **Validation** : Vérifie que le message contient `asset_id` et `features`
3. **Conversion** : Convertit le message en `RULPredictionRequest`
4. **Prédiction** : Appelle `rul_prediction_service.predict_rul()`
5. **Publication** : Publie le résultat sur le topic `rul-predictions`
6. **Logging** : Log les prédictions importantes

## Gestion des Erreurs

### Modèles Non Entraînés
- Le worker démarre même si les modèles ne sont pas entraînés
- Les messages sont ignorés avec un warning
- Entraîner les modèles via l'API pour activer les prédictions

### Messages Invalides
- Messages sans `asset_id` : ignorés
- Messages sans `features` : ignorés
- Messages JSON invalides : ignorés avec warning

### Erreurs de Prédiction
- Erreurs loggées mais ne bloquent pas le worker
- Le worker continue à traiter les messages suivants

## Arrêt Propre

Le worker gère les signaux SIGINT et SIGTERM pour un arrêt propre :

```bash
# Arrêt avec Ctrl+C
# Ou
kill -SIGTERM <pid>
```

Lors de l'arrêt :
- Les messages en cours sont traités
- Les connexions Kafka sont fermées proprement
- Les ressources sont libérées

## Monitoring

### Logs

Le worker log :
- Démarrage/arrêt
- Messages traités
- Erreurs
- Prédictions importantes

**Niveau de log** : Configurable via `LOG_LEVEL` dans la config

### Métriques

Pour monitorer le worker :
- Nombre de messages consommés
- Nombre de prédictions publiées
- Taux d'erreur
- Latence de prédiction

## Exemples

### Exemple 1 : Démarrer le Worker

```bash
# Terminal 1 : Démarrer Kafka (si local)
docker-compose up kafka

# Terminal 2 : Démarrer le worker
cd services/prediction-rul
python -m app.worker
```

### Exemple 2 : Tester avec un Message

```bash
# Publier un message de test
kafka-console-producer \
  --bootstrap-server localhost:9092 \
  --topic extracted-features \
  <<< '{
    "asset_id": "ASSET001",
    "sensor_id": "SENSOR001",
    "features": {"rms": 10.5, "kurtosis": 2.3},
    "timestamp": "2024-01-01T12:00:00Z"
  }'

# Consulter les prédictions
kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic rul-predictions \
  --from-beginning
```

## Troubleshooting

### Worker ne démarre pas
- Vérifier que Kafka est accessible
- Vérifier la configuration Kafka dans `.env`
- Vérifier les logs pour les erreurs

### Aucune prédiction publiée
- Vérifier que les modèles sont entraînés
- Vérifier que des messages arrivent sur `extracted-features`
- Vérifier les logs pour les erreurs

### Erreurs de connexion Kafka
- Vérifier `kafka_bootstrap_servers`
- Vérifier que Kafka est démarré
- Vérifier les permissions réseau

### Messages non traités
- Vérifier le format des messages
- Vérifier que `asset_id` et `features` sont présents
- Vérifier les logs pour les warnings

## Performance

### Optimisations
- **Batch processing** : Traiter plusieurs messages en batch (à implémenter)
- **Async processing** : Utiliser asyncio pour traitement parallèle
- **Caching** : Cacher les prédictions récentes

### Limitations
- Traitement séquentiel (un message à la fois)
- Pas de retry automatique en cas d'erreur
- Pas de backpressure handling

## Sécurité

- **Authentification Kafka** : Supporté via configuration
- **SSL/TLS** : Supporté via configuration
- **Validation** : Messages validés avant traitement

## Notes

1. Le worker est conçu pour tourner en continu
2. Les modèles doivent être entraînés avant de démarrer
3. Le worker gère automatiquement les reconnexions Kafka
4. Les prédictions sont publiées même si le modèle n'est pas calibré

