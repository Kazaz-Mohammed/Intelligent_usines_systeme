# Service DétectionAnomalies

## Description

Service Python/FastAPI avec modèles ML/DL custom pour la détection d'anomalies en temps-réel.

## Fonctionnalités

- Modèles PyOD (IsolationForest, One-Class SVM)
- Autoencodeurs LSTM custom (PyTorch)
- Scoring temps-réel
- Seuils adaptatifs par criticité
- Journalisation des événements
- Consumer Kafka pour features
- MLflow tracking

## Technologies

- Python/FastAPI
- PyOD
- PyTorch (modèles custom)
- MLflow
- PostgreSQL
- Kafka

## Structure

```
detection-anomalies/
├── app/
│   ├── __init__.py
│   ├── main.py                      # Point d'entrée FastAPI
│   ├── config.py                    # Configuration
│   ├── worker.py                    # Worker Kafka pour traitement temps-réel
│   ├── models/                      # Modèles de données Pydantic
│   │   └── anomaly_data.py
│   ├── services/                    # Services métier
│   │   ├── anomaly_detection_service.py  # Orchestration des modèles
│   │   ├── isolation_forest_service.py   # Isolation Forest (PyOD)
│   │   ├── one_class_svm_service.py      # One-Class SVM (PyOD)
│   │   ├── lstm_autoencoder_service.py   # LSTM Autoencoder (PyTorch)
│   │   ├── kafka_consumer.py             # Consumer Kafka
│   │   ├── kafka_producer.py             # Producer Kafka
│   │   └── mlflow_service.py             # Service MLflow
│   ├── api/                         # Endpoints FastAPI
│   │   └── anomalies.py
│   └── database/                    # Accès base de données
│       └── postgresql.py            # Service PostgreSQL pour journalisation
├── tests/                           # Tests unitaires et intégration
│   ├── test_main.py
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_isolation_forest_service.py
│   ├── test_one_class_svm_service.py
│   ├── test_lstm_autoencoder_service.py
│   ├── test_anomaly_detection_service.py
│   ├── test_api_anomalies.py
│   ├── test_kafka_consumer.py
│   ├── test_kafka_producer.py
│   ├── test_mlflow_service.py
│   └── test_postgresql_service.py
├── requirements.txt
├── pytest.ini
├── README.md
└── PHASE_6_PROGRESS.md
```

## État

✅ **Phase 6 complète - Tous les composants implémentés** (7/7 composants)

## Fonctionnalités implémentées

### ✅ Modèles de détection d'anomalies
- **Isolation Forest** (PyOD) - Détection d'anomalies basée sur l'isolement
- **One-Class SVM** (PyOD) - SVM pour détection d'anomalies
- **LSTM Autoencoder** (PyTorch) - Autoencodeur LSTM pour détection d'anomalies temporelles

### ✅ API REST FastAPI
- `POST /api/v1/anomalies/detect` - Détecter une anomalie (une requête)
- `POST /api/v1/anomalies/detect/batch` - Détecter des anomalies (batch)
- `POST /api/v1/anomalies/train` - Entraîner les modèles
- `GET /api/v1/anomalies/status` - Statut des modèles (entraînés ou non)
- `GET /api/v1/anomalies/` - Historique des anomalies depuis PostgreSQL (avec filtres et pagination)
- `GET /api/v1/anomalies/metrics` - Métriques MLflow (placeholder)

### ✅ Worker Kafka
- Consomme les features depuis le topic `extracted-features`
- Détecte les anomalies en temps-réel
- Publie les résultats sur le topic `anomalies-detected`
- Journalise automatiquement les anomalies dans PostgreSQL
- Gestion des erreurs et reconnexion automatique

### ✅ MLflow Integration
- Tracking des expériences d'entraînement
- Registry des modèles
- Logging des paramètres et métriques
- Support sklearn et PyTorch
- Chargement de modèles depuis le registry

### ✅ Journalisation PostgreSQL
- Service PostgreSQL avec pool de connexions
- Table `anomaly_detections` avec index optimisés
- Journalisation automatique des anomalies détectées
- Endpoint GET `/api/v1/anomalies/` avec filtres et pagination
- Support des filtres : asset_id, sensor_id, dates, is_anomaly, criticality

## Installation

```bash
# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Démarrer l'API FastAPI

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8084
```

L'API sera disponible sur `http://localhost:8084`

### Démarrer le Worker Kafka

```bash
python -m app.worker
```

Le worker consomme automatiquement les features depuis Kafka et publie les anomalies détectées.

**Important** : Les modèles doivent être entraînés avant de démarrer le worker. Utilisez l'endpoint `POST /api/v1/anomalies/train` pour entraîner les modèles.

### Configuration

Les paramètres peuvent être configurés via variables d'environnement ou fichier `.env` :

```bash
# Service
SERVICE_NAME=detection-anomalies-service
SERVICE_PORT=8084
LOG_LEVEL=INFO

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_CONSUMER_GROUP=detection-anomalies-service
KAFKA_TOPIC_INPUT_FEATURES=extracted-features
KAFKA_TOPIC_OUTPUT_ANOMALIES=anomalies-detected

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=anomaly-detection
MLFLOW_ENABLED=true

# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=predictive_maintenance
DATABASE_USER=pmuser
DATABASE_PASSWORD=pmpassword
```

## Tests

```bash
# Lancer tous les tests
pytest

# Lancer avec coverage
pytest --cov=app --cov-report=html

# Lancer un test spécifique
pytest tests/test_anomaly_detection_service.py -v
```

**83 tests passent** (100%) ✅

## Architecture

```
┌─────────────────┐
│  Kafka Topics   │
│                 │
│ extracted-      │──┐
│   features      │  │
└─────────────────┘  │
                     │ Consomme
                     ▼
            ┌────────────────┐
            │ Kafka Consumer │
            └────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ Anomaly Detection      │
        │ Service                │
        │                        │
        │  ┌──────────────────┐  │
        │  │ Isolation Forest │  │
        │  └──────────────────┘  │
        │  ┌──────────────────┐  │
        │  │ One-Class SVM    │  │
        │  └──────────────────┘  │
        │  ┌──────────────────┐  │
        │  │ LSTM Autoencoder │  │
        │  └──────────────────┘  │
        └────────────────────────┘
                     │
                     │ Détecte
                     ▼
            ┌────────────────┐
            │ Kafka Producer │
            └────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Kafka Topics          │
        │                        │
        │ anomalies-detected     │
        └────────────────────────┘

        ┌────────────────────────┐
        │  FastAPI REST API      │
        │                        │
        │  /detect               │
        │  /train                │
        │  /status               │
        └────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  MLflow                │
        │                        │
        │  Tracking              │
        │  Registry              │
        └────────────────────────┘
```

## Exemples d'utilisation

### Récupérer les anomalies

```bash
# Toutes les anomalies
curl "http://localhost:8084/api/v1/anomalies/"

# Filtrer par asset_id
curl "http://localhost:8084/api/v1/anomalies/?asset_id=ASSET001"

# Filtrer par criticité
curl "http://localhost:8084/api/v1/anomalies/?criticality=high"

# Filtrer par date
curl "http://localhost:8084/api/v1/anomalies/?start_date=2024-01-01T00:00:00Z&end_date=2024-01-31T23:59:59Z"

# Pagination
curl "http://localhost:8084/api/v1/anomalies/?limit=50&offset=0"
```

## Prochaines étapes

- [ ] Dockerfile et docker-compose.yml
- [ ] Documentation des API avec exemples détaillés

