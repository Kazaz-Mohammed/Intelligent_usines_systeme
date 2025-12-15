# Guide du Worker Kafka

## Description

Le worker Kafka (`app/worker.py`) est un service en arrière-plan qui consomme les features extraites depuis Kafka, détecte les anomalies en temps-réel, et publie les résultats sur Kafka.

## Architecture

```
┌─────────────────────┐
│  Kafka Topic        │
│  extracted-features │
└──────────┬──────────┘
           │
           │ Consomme
           ▼
┌─────────────────────┐
│  Kafka Consumer     │
│  Service            │
└──────────┬──────────┘
           │
           │ Traite
           ▼
┌─────────────────────┐
│  Anomaly Detection  │
│  Service            │
│                     │
│  - Isolation Forest │
│  - One-Class SVM    │
│  - LSTM Autoencoder │
└──────────┬──────────┘
           │
           │ Résultats
           ▼
┌─────────────────────┐
│  Kafka Producer     │
│  Service            │
└──────────┬──────────┘
           │
           │ Publie
           ▼
┌─────────────────────┐
│  Kafka Topic        │
│  anomalies-detected │
└─────────────────────┘
```

## Démarrage

### Prérequis

1. **Kafka doit être démarré** avec les topics suivants :
   - `extracted-features` (input)
   - `anomalies-detected` (output)

2. **Les modèles doivent être entraînés** avant de démarrer le worker.

### Démarrer le worker

```bash
# Méthode 1 : Directement
python -m app.worker

# Méthode 2 : Avec uvicorn (si besoin)
python app/worker.py
```

### Vérifier que les modèles sont entraînés

Avant de démarrer le worker, vérifiez le statut des modèles via l'API :

```bash
curl http://localhost:8084/api/v1/anomalies/status
```

Si les modèles ne sont pas entraînés, entraînez-les d'abord :

```bash
curl -X POST "http://localhost:8084/api/v1/anomalies/train" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[...], [...]],
    "feature_names": ["rms", "kurtosis", ...]
  }'
```

## Format des messages

### Input (extracted-features)

Le worker s'attend à recevoir des messages JSON avec la structure suivante :

```json
{
  "asset_id": "ASSET001",
  "sensor_id": "SENSOR001",
  "features": {
    "rms": 0.5,
    "kurtosis": 0.3,
    "crest_factor": 1.2,
    "variance": 0.1,
    "mean": 0.0
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "metadata": {}
}
```

### Output (anomalies-detected)

Le worker publie des messages JSON avec la structure suivante :

```json
{
  "asset_id": "ASSET001",
  "sensor_id": "SENSOR001",
  "timestamp": "2024-01-01T12:00:00Z",
  "scores": [
    {
      "score": 0.75,
      "model_name": "isolation_forest",
      "threshold": 0.1,
      "is_anomaly": true
    },
    {
      "score": 0.68,
      "model_name": "one_class_svm",
      "threshold": 0.1,
      "is_anomaly": true
    },
    {
      "score": 0.82,
      "model_name": "lstm_autoencoder",
      "threshold": 0.95,
      "is_anomaly": true
    }
  ],
  "final_score": 0.75,
  "is_anomaly": true,
  "criticality": "high",
  "features": {
    "rms": 0.5,
    "kurtosis": 0.3,
    "crest_factor": 1.2,
    "variance": 0.1,
    "mean": 0.0
  },
  "metadata": {}
}
```

## Configuration

### Variables d'environnement

```bash
# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_CONSUMER_GROUP=detection-anomalies-service
KAFKA_TOPIC_INPUT_FEATURES=extracted-features
KAFKA_TOPIC_OUTPUT_ANOMALIES=anomalies-detected
KAFKA_AUTO_OFFSET_RESET=earliest
KAFKA_ENABLE_AUTO_COMMIT=true

# Service
LOG_LEVEL=INFO
```

### Dans le code

Voir `app/config.py` pour toutes les options de configuration.

## Gestion des erreurs

Le worker gère automatiquement :

- **Messages JSON invalides** : Loggés comme erreurs et ignorés
- **Messages sans asset_id ou features** : Loggés comme warnings et ignorés
- **Modèles non entraînés** : Avertissement affiché au démarrage, messages ignorés
- **Erreurs Kafka** : Reconnexion automatique
- **Signaux d'arrêt** : SIGINT et SIGTERM sont gérés proprement

## Logs

Le worker log :

- **INFO** : Démarrage/arrêt du worker, abonnement aux topics
- **WARNING** : Anomalies détectées avec leur criticité
- **DEBUG** : Pas d'anomalie détectée, features publiées
- **ERROR** : Erreurs lors du traitement, messages invalides

Exemple de log :

```
INFO - Démarrage du worker de détection d'anomalies...
INFO - Abonné au topic: extracted-features
INFO - Thread de consommation démarré
WARNING - Anomalie détectée: ASSET001 (score=0.823, criticality=high)
DEBUG - Pas d'anomalie: ASSET002 (score=0.123)
```

## Arrêt propre

Le worker peut être arrêté proprement avec :

- **Ctrl+C** (SIGINT)
- **kill -TERM <pid>** (SIGTERM)

Le worker fermera automatiquement les connexions Kafka avant de s'arrêter.

## Exemples d'utilisation

### Démarrer le worker en arrière-plan

```bash
# Linux/Mac
python -m app.worker > worker.log 2>&1 &

# Windows PowerShell
Start-Process python -ArgumentList "-m", "app.worker" -RedirectStandardOutput worker.log -RedirectStandardError worker.err
```

### Vérifier que le worker fonctionne

```bash
# Vérifier les logs
tail -f worker.log

# Vérifier qu'il consomme des messages
# (nécessite Kafka en fonctionnement avec des messages dans le topic)
```

### Arrêter le worker

```bash
# Trouver le PID
ps aux | grep "app.worker"

# Arrêter proprement
kill -TERM <pid>
```

## Troubleshooting

### Le worker ne détecte pas d'anomalies

1. **Vérifier que les modèles sont entraînés** :
   ```bash
   curl http://localhost:8084/api/v1/anomalies/status
   ```

2. **Vérifier les logs** : Chercher des messages comme "Aucun modèle n'est entraîné"

3. **Vérifier que Kafka reçoit des messages** : Utiliser `kafka-console-consumer` pour vérifier le topic `extracted-features`

### Le worker ne démarre pas

1. **Vérifier que Kafka est accessible** :
   ```bash
   # Vérifier que les topics existent
   kafka-topics --list --bootstrap-server localhost:9092
   ```

2. **Vérifier la configuration** : Vérifier les variables d'environnement ou le fichier `.env`

3. **Vérifier les logs d'erreur** : Le worker affichera les erreurs de connexion Kafka

### Messages ignorés

Si des messages sont ignorés, vérifier :

1. **Format JSON valide** : Le message doit être un JSON valide
2. **Champs requis** : `asset_id` et `features` sont requis
3. **Format des features** : `features` doit être un dictionnaire {nom: valeur}

## Performance

Le worker traite les messages un par un. Pour améliorer les performances :

- Utiliser `consume_features_batch` pour traiter plusieurs messages à la fois (à implémenter)
- Augmenter le nombre de workers parallèles
- Optimiser les modèles (réduction de la complexité)

## Intégration avec d'autres services

Le worker s'intègre naturellement avec :

- **Service Extraction Features** : Consomme les features qu'il publie
- **Service de Notification** : Peut consommer `anomalies-detected` pour notifier les utilisateurs
- **Dashboard** : Peut consommer `anomalies-detected` pour afficher en temps-réel

