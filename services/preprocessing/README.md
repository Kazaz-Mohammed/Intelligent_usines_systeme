# Service Prétraitement

## Description

Service Python/FastAPI responsable du nettoyage et de la normalisation des données capteurs avant analyse ML.

## Fonctionnalités

- ✅ Consumer Kafka (topic `sensor-data`)
- ⏳ Nettoyage des données (détection valeurs aberrantes)
- ⏳ Rééchantillonnage et synchronisation multi-capteurs
- ⏳ Débruitage (filtres passe-bande)
- ⏳ Analyse fréquentielle (STFT/FFT)
- ⏳ Fenêtrage glissant pour ML
- ⏳ Producer Kafka (topic `preprocessed-data`)
- ⏳ Stockage dans TimescaleDB

## Technologies

- **Python 3.11+**
- **FastAPI** (API REST)
- **Pandas** (manipulation données)
- **SciPy** (traitement signal)
- **NumPy** (calculs numériques)
- **Apache Kafka** (confluent-kafka)
- **TimescaleDB** (psycopg2)
- **Pydantic** (validation données)

## Structure du Projet

```
preprocessing/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration
│   ├── models/                 # Modèles de données
│   │   ├── __init__.py
│   │   └── sensor_data.py
│   ├── services/               # Services métier
│   │   ├── __init__.py
│   │   ├── kafka_consumer.py
│   │   ├── kafka_producer.py
│   │   ├── cleaning_service.py
│   │   ├── resampling_service.py
│   │   ├── denoising_service.py
│   │   ├── frequency_analysis_service.py
│   │   └── windowing_service.py
│   ├── database/              # Accès base de données
│   │   ├── __init__.py
│   │   └── timescaledb.py
│   └── api/                   # Endpoints REST
│       ├── __init__.py
│       └── preprocessing.py
├── tests/                     # Tests
│   ├── __init__.py
│   ├── test_cleaning.py
│   ├── test_resampling.py
│   ├── test_denoising.py
│   └── test_integration.py
├── requirements.txt           # Dépendances Python
├── Dockerfile                 # Image Docker
├── .dockerignore
└── README.md
```

## Configuration

### Variables d'Environnement

```bash
# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_CONSUMER_GROUP=preprocessing-service
KAFKA_TOPIC_INPUT=sensor-data
KAFKA_TOPIC_OUTPUT=preprocessed-data

# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=predictive_maintenance
DATABASE_USER=pmuser
DATABASE_PASSWORD=pmpassword

# Service
SERVICE_PORT=8082
LOG_LEVEL=INFO
```

## Démarrage

### Local (avec Python)

```bash
# Installer les dépendances
pip install -r requirements.txt

# Démarrer le service
uvicorn app.main:app --host 0.0.0.0 --port 8082 --reload
```

### Docker

```bash
# Build
docker build -t preprocessing-service:latest .

# Run
docker run -p 8082:8082 \
  -e KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  -e DATABASE_HOST=localhost \
  preprocessing-service:latest
```

## API REST

### Health Check
```
GET /api/v1/preprocessing/health
```

### Status
```
GET /api/v1/preprocessing/status
```

### Metrics
```
GET /api/v1/preprocessing/metrics
```

## Tests

```bash
# Tous les tests
pytest

# Avec couverture
pytest --cov=app --cov-report=html

# Tests spécifiques
pytest tests/test_cleaning.py
```

## État

🚧 **En développement** (Phase 3)

