# Guide Utilisateur - Service Prediction RUL

## Vue d'ensemble

Le service **Prediction RUL** (Remaining Useful Life) prédit la durée de vie restante des équipements industriels en utilisant des modèles de machine learning avancés (LSTM, GRU, TCN, XGBoost).

## Fonctionnalités Principales

### ✅ Prédiction RUL en Temps-Réel
- Prédiction de la RUL pour un actif donné
- Support de plusieurs modèles (LSTM, GRU, TCN, XGBoost)
- Agrégation d'ensemble pour améliorer la précision
- Intervalles de confiance pour quantifier l'incertitude

### ✅ Calibration des Modèles
- Calibration isotonique
- Platt Scaling
- Temperature Scaling
- Amélioration de la précision des prédictions

### ✅ Transfer Learning
- Pré-entraînement sur NASA C-MAPSS
- Fine-tuning sur données spécifiques
- Gel de couches pour préserver les connaissances

### ✅ Journalisation PostgreSQL
- Historique complet des prédictions
- Requêtes avec filtres avancés
- Analyse de tendances

### ✅ Intégration Kafka
- Consommation de features en temps-réel
- Publication des prédictions RUL
- Traitement asynchrone

### ✅ MLflow Tracking
- Suivi des expériences d'entraînement
- Registre des modèles
- Comparaison de performances

## Démarrage Rapide

### 1. Installation

```bash
cd services/prediction-rul
pip install -r requirements.txt
```

### 2. Configuration

Créer un fichier `.env` :

```bash
# Service
SERVICE_NAME=prediction-rul-service
SERVICE_PORT=8085
LOG_LEVEL=INFO

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_CONSUMER_GROUP=prediction-rul-service
KAFKA_TOPIC_INPUT_FEATURES=extracted-features
KAFKA_TOPIC_OUTPUT_RUL=rul-predictions

# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=predictive_maintenance
DATABASE_USER=pmuser
DATABASE_PASSWORD=pmpassword

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=rul-prediction
MLFLOW_ENABLED=true
```

### 3. Démarrer le Service

```bash
# Service API REST
uvicorn app.main:app --host 0.0.0.0 --port 8085

# Worker Kafka (optionnel, pour traitement temps-réel)
python app/worker.py
```

## Utilisation de l'API REST

### Prédire la RUL

```bash
POST /api/v1/rul/predict
```

**Requête :**
```json
{
    "asset_id": "PUMP_001",
    "sensor_id": "SENSOR_001",
    "features": {
        "rms": 10.5,
        "kurtosis": 2.3,
        "skewness": 0.8,
        "peak": 15.2,
        "crest_factor": 3.5
    },
    "sequence_data": [
        {"rms": 10.0, "kurtosis": 2.0},
        {"rms": 10.2, "kurtosis": 2.1},
        {"rms": 10.5, "kurtosis": 2.3}
    ],
    "timestamp": "2024-01-15T10:30:00Z"
}
```

**Réponse :**
```json
{
    "asset_id": "PUMP_001",
    "sensor_id": "SENSOR_001",
    "timestamp": "2024-01-15T10:30:00Z",
    "rul_prediction": 150.5,
    "confidence_interval_lower": 140.0,
    "confidence_interval_upper": 160.0,
    "confidence_level": 0.95,
    "uncertainty": 10.0,
    "model_used": "ensemble",
    "model_scores": {
        "lstm": 150.0,
        "gru": 151.0,
        "tcn": 149.5,
        "xgboost": 151.5
    },
    "features": {
        "rms": 10.5,
        "kurtosis": 2.3
    },
    "metadata": {}
}
```

### Prédiction Batch

```bash
POST /api/v1/rul/predict/batch
```

**Requête :**
```json
[
    {
        "asset_id": "PUMP_001",
        "features": {"rms": 10.5},
        "sequence_data": [...]
    },
    {
        "asset_id": "PUMP_002",
        "features": {"rms": 12.0},
        "sequence_data": [...]
    }
]
```

### Récupérer l'Historique

```bash
GET /api/v1/rul/?asset_id=PUMP_001&limit=100&offset=0
```

**Paramètres :**
- `asset_id` : Filtrer par actif
- `sensor_id` : Filtrer par capteur
- `start_date` : Date de début (ISO format)
- `end_date` : Date de fin (ISO format)
- `model_used` : Filtrer par modèle (lstm, gru, tcn, xgboost, ensemble)
- `limit` : Nombre de résultats (1-1000)
- `offset` : Offset pour pagination

### Statut du Service

```bash
GET /api/v1/rul/status
```

**Réponse :**
```json
{
    "ready": true,
    "models": {
        "lstm": {
            "is_trained": true,
            "input_size": 5,
            "hidden_size": 128
        },
        "gru": {
            "is_trained": true
        },
        "tcn": {
            "is_trained": true
        },
        "xgboost": {
            "is_trained": true
        }
    },
    "calibration": {
        "lstm": {
            "is_calibrated": true,
            "method": "isotonic"
        }
    },
    "transfer_learning": {
        "lstm": {
            "is_pretrained_loaded": true,
            "pretrained_path": "/models/lstm_pretrained.pth"
        }
    },
    "mlflow": {
        "enabled": true,
        "tracking_uri": "http://localhost:5000",
        "experiment_name": "rul-prediction"
    }
}
```

## Entraînement des Modèles

### Entraîner Tous les Modèles

```bash
POST /api/v1/rul/train
```

**Requête :**
```json
{
    "training_data": [
        [10.5, 2.3, 0.8, 15.2, 3.5],
        [11.0, 2.5, 0.9, 16.0, 3.6],
        ...
    ],
    "target_data": [150.0, 145.0, 140.0, ...],
    "feature_names": ["rms", "kurtosis", "skewness", "peak", "crest_factor"],
    "parameters": {
        "epochs": 100,
        "batch_size": 32
    },
    "use_transfer_learning": true
}
```

### Entraîner un Modèle Spécifique

```json
{
    "model_name": "lstm",
    "training_data": [...],
    "target_data": [...],
    "parameters": {
        "epochs": 50
    }
}
```

## Calibration

### Calibrer les Modèles

```bash
POST /api/v1/rul/calibrate
```

**Requête :**
```json
{
    "predictions": [150.0, 145.0, 140.0, ...],
    "actuals": [148.0, 143.0, 138.0, ...],
    "method": "isotonic",
    "model_name": "lstm"
}
```

**Méthodes disponibles :**
- `isotonic` : Régression isotonique (recommandé)
- `platt` : Platt Scaling
- `temperature_scaling` : Temperature Scaling

## Transfer Learning

### Charger un Modèle Pré-entraîné

```bash
POST /api/v1/rul/transfer-learning/load
```

**Requête :**
```json
{
    "model_name": "lstm",
    "path": "/models/lstm_pretrained_nasa_cmapss.pth",
    "freeze_layers": false
}
```

### Informations sur le Transfer Learning

```bash
GET /api/v1/rul/transfer-learning/info/lstm
```

## Worker Kafka

### Démarrer le Worker

```bash
python app/worker.py
```

Le worker :
1. Consomme les features depuis `extracted-features`
2. Prédit la RUL pour chaque message
3. Publie les prédictions sur `rul-predictions`
4. Journalise dans PostgreSQL

### Format des Messages

**Input (extracted-features) :**
```json
{
    "asset_id": "PUMP_001",
    "sensor_id": "SENSOR_001",
    "timestamp": "2024-01-15T10:30:00Z",
    "features": [
        {
            "feature_name": "rms",
            "feature_value": 10.5,
            "feature_type": "temporal"
        },
        {
            "feature_name": "kurtosis",
            "feature_value": 2.3,
            "feature_type": "temporal"
        }
    ]
}
```

**Output (rul-predictions) :**
```json
{
    "asset_id": "PUMP_001",
    "sensor_id": "SENSOR_001",
    "timestamp": "2024-01-15T10:30:00Z",
    "rul_prediction": 150.5,
    "confidence_interval_lower": 140.0,
    "confidence_interval_upper": 160.0,
    "model_used": "ensemble",
    "model_scores": {...}
}
```

## Exemples d'Utilisation

### Exemple 1 : Prédiction Simple

```python
import requests

response = requests.post(
    "http://localhost:8085/api/v1/rul/predict",
    json={
        "asset_id": "PUMP_001",
        "features": {
            "rms": 10.5,
            "kurtosis": 2.3
        },
        "sequence_data": [
            {"rms": 10.0, "kurtosis": 2.0},
            {"rms": 10.2, "kurtosis": 2.1},
            {"rms": 10.5, "kurtosis": 2.3}
        ]
    }
)

result = response.json()
print(f"RUL prédite: {result['rul_prediction']} cycles")
print(f"Intervalle de confiance: [{result['confidence_interval_lower']}, {result['confidence_interval_upper']}]")
```

### Exemple 2 : Entraînement avec Transfer Learning

```python
import requests
import numpy as np

# Charger le modèle pré-entraîné
requests.post(
    "http://localhost:8085/api/v1/rul/transfer-learning/load",
    json={
        "model_name": "lstm",
        "path": "/models/lstm_pretrained_nasa_cmapss.pth",
        "freeze_layers": False
    }
)

# Entraîner avec données spécifiques
X_train = np.random.randn(1000, 10, 5).tolist()
y_train = np.random.rand(1000) * 200

response = requests.post(
    "http://localhost:8085/api/v1/rul/train",
    json={
        "model_name": "lstm",
        "training_data": X_train,
        "target_data": y_train.tolist(),
        "use_transfer_learning": True
    }
)

print(response.json())
```

### Exemple 3 : Calibration

```python
import requests

# Obtenir des prédictions de validation
predictions = [150.0, 145.0, 140.0]
actuals = [148.0, 143.0, 138.0]

response = requests.post(
    "http://localhost:8085/api/v1/rul/calibrate",
    json={
        "predictions": predictions,
        "actuals": actuals,
        "method": "isotonic",
        "model_name": "lstm"
    }
)

print(response.json())
```

## Interprétation des Résultats

### RUL Prediction
- **Valeur** : Durée de vie restante en cycles ou heures
- **Interprétation** : Plus la valeur est élevée, plus l'équipement a de temps avant la maintenance

### Confidence Interval
- **Intervalle** : [lower, upper]
- **Interprétation** : 95% de confiance que la vraie RUL se trouve dans cet intervalle
- **Plus l'intervalle est étroit, plus la prédiction est fiable**

### Uncertainty
- **Valeur** : Incertitude de la prédiction
- **Interprétation** : Plus la valeur est faible, plus la prédiction est certaine

### Model Scores
- **Scores individuels** : Prédictions de chaque modèle
- **Ensemble** : Agrégation des prédictions pour améliorer la robustesse

## Bonnes Pratiques

1. **Séquences Temporelles** : Fournir des séquences de longueur suffisante (≥ sequence_length du modèle)
2. **Calibration** : Calibrer les modèles avec des données de validation
3. **Transfer Learning** : Utiliser des modèles pré-entraînés pour améliorer les performances
4. **Monitoring** : Surveiller les métriques MLflow pour suivre les performances
5. **Historique** : Utiliser PostgreSQL pour analyser les tendances de RUL

## Troubleshooting

### Service non prêt
```
RuntimeError: Aucun modèle n'est entraîné
```
**Solution** : Entraîner les modèles via `POST /api/v1/rul/train`

### Erreur de séquence
```
ValueError: Séquence trop courte
```
**Solution** : Fournir une séquence de longueur ≥ sequence_length (par défaut 20)

### Erreur PostgreSQL
```
ConnectionError: Le pool de connexions PostgreSQL n'est pas initialisé
```
**Solution** : Vérifier la configuration de la base de données dans `.env`

## Ressources

- [MLflow Guide](MLFLOW_GUIDE.md)
- [PostgreSQL Guide](POSTGRESQL_GUIDE.md)
- [Transfer Learning Guide](TRANSFER_LEARNING_GUIDE.md)
- [Worker Kafka Guide](WORKER_KAFKA.md)

