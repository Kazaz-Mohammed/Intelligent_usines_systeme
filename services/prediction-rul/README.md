# Service Prediction RUL

## Description

Service Python/FastAPI pour la prédiction de la Remaining Useful Life (RUL) en temps-réel.

## Fonctionnalités

- Modèles LSTM/GRU/TCN pour prédiction RUL
- XGBoost comme baseline
- Transfer learning depuis NASA C-MAPSS
- Calibration et quantification d'incertitude
- Intervalles de confiance
- Caching des prédictions
- MLflow tracking

## Technologies

- Python/FastAPI
- PyTorch (LSTM, GRU, TCN)
- XGBoost
- MLflow
- PostgreSQL
- Kafka

## Structure

```
prediction-rul/
├── app/
│   ├── __init__.py
│   ├── main.py                      # Point d'entrée FastAPI
│   ├── config.py                    # Configuration
│   ├── models/                      # Modèles de données Pydantic
│   │   └── rul_data.py
│   ├── services/                    # Services métier
│   │   ├── lstm_service.py          # Service LSTM
│   │   ├── gru_service.py           # Service GRU
│   │   ├── tcn_service.py           # Service TCN
│   │   ├── xgboost_service.py      # Service XGBoost
│   │   ├── rul_prediction_service.py # Orchestration
│   │   └── calibration_service.py  # Calibration
│   ├── api/                         # Endpoints FastAPI
│   │   └── rul.py
│   └── database/                    # Accès base de données
│       └── postgresql.py
├── tests/                           # Tests unitaires et intégration
├── requirements.txt
├── pytest.ini
└── README.md
```

## État

✅ **Phase 7 complète - Tous les composants implémentés** (10/10 composants)

## Installation

```bash
# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Démarrer l'API FastAPI

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8085
```

L'API sera disponible sur `http://localhost:8085`

## Configuration

Les paramètres peuvent être configurés via variables d'environnement ou fichier `.env` :

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

## Tests

```bash
# Lancer tous les tests
pytest

# Lancer avec coverage
pytest --cov=app --cov-report=html
```

## Fonctionnalités implémentées

### ✅ Modèles RUL
- **LSTM** - Réseau de neurones récurrents LSTM
- **GRU** - Réseau de neurones récurrents GRU
- **TCN** - Temporal Convolutional Network
- **XGBoost** - Gradient boosting (baseline)

### ✅ Service de Prédiction Principal
- Orchestration des 4 modèles
- Entraînement de tous les modèles
- Prédiction avec agrégation (ensemble)
- Prédiction avec un modèle spécifique
- Prédiction batch
- Gestion du statut des modèles

### ✅ API REST FastAPI
- `POST /api/v1/rul/predict` - Prédire la RUL
- `POST /api/v1/rul/predict/batch` - Prédiction batch
- `POST /api/v1/rul/train` - Entraîner les modèles
- `POST /api/v1/rul/calibrate` - Calibrer les modèles
- `POST /api/v1/rul/transfer-learning/load` - Charger un modèle pré-entraîné
- `GET /api/v1/rul/transfer-learning/info` - Informations sur le transfer learning
- `GET /api/v1/rul/status` - Statut des modèles, calibration et transfer learning
- `GET /api/v1/rul/` - Historique (placeholder)

### ✅ Calibration et Incertitude
- Service de calibration avec 3 méthodes (isotonic, platt, temperature_scaling)
- Quantification d'incertitude (std, quantile, ensemble)
- Intervalles de confiance (90%, 95%, 99%)
- Intégration automatique dans les prédictions

### ✅ Transfer Learning
- Service de transfer learning depuis NASA C-MAPSS
- Chargement de modèles pré-entraînés (LSTM, GRU, TCN)
- Fine-tuning avec learning rate réduit
- Gel des couches de l'encodeur (optionnel)
- Intégration automatique dans les services

### ✅ Kafka Integration
- Consumer Kafka pour consommer les features
- Producer Kafka pour publier les prédictions RUL
- Worker Kafka pour traitement temps-réel
- Gestion des erreurs et reconnexions

### ✅ MLflow Service
- Tracking des expériences d'entraînement
- Registre des modèles (LSTM, GRU, TCN, XGBoost)
- Logging automatique des paramètres et métriques
- Chargement de modèles depuis le registry
- Recherche et comparaison de runs

### ✅ PostgreSQL Logging
- Journalisation automatique des prédictions RUL
- Table `rul_predictions` avec index optimisés
- API REST pour récupérer l'historique
- Intégration Worker Kafka

### ✅ Tests
- 100 tests passent (100%)

## Documentation

- [User Guide](USER_GUIDE.md) - Guide complet d'utilisation
- [Architecture](ARCHITECTURE.md) - Architecture détaillée du service
- [Phase 7 Progress](PHASE_7_PROGRESS.md) - Progression de la phase 7
- [MLflow Guide](MLFLOW_GUIDE.md) - Guide MLflow
- [PostgreSQL Guide](POSTGRESQL_GUIDE.md) - Guide PostgreSQL
- [Transfer Learning Guide](TRANSFER_LEARNING_GUIDE.md) - Guide Transfer Learning
- [Worker Kafka Guide](WORKER_KAFKA.md) - Guide Worker Kafka
