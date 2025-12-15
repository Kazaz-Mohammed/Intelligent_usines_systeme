# Architecture - Service Prediction RUL

## Vue d'ensemble

Le service **Prediction RUL** est un microservice FastAPI qui prédit la durée de vie restante (RUL) des équipements industriels en utilisant des modèles de machine learning avancés.

## Architecture Générale

```
┌─────────────────────────────────────────────────────────────┐
│                    Prediction RUL Service                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   FastAPI    │  │   Kafka       │  │  PostgreSQL  │     │
│  │   REST API   │  │   Worker     │  │  Logging     │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │              │
│  ┌──────▼─────────────────▼─────────────────▼──────┐     │
│  │         RUL Prediction Service                  │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │     │
│  │  │   LSTM   │  │   GRU    │  │   TCN    │       │     │
│  │  └──────────┘  └──────────┘  └──────────┘       │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │     │
│  │  │ XGBoost  │  │Calibration│  │Transfer  │       │     │
│  │  │          │  │  Service  │  │ Learning │       │     │
│  │  └──────────┘  └──────────┘  └──────────┘       │     │
│  └──────────────────────────────────────────────────┘     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                      │
│  │   MLflow     │  │   Kafka      │                      │
│  │   Tracking   │  │  Producer    │                      │
│  └──────────────┘  └──────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

## Composants Principaux

### 1. API REST (FastAPI)

**Fichier** : `app/api/rul.py`

**Endpoints :**
- `POST /api/v1/rul/predict` - Prédiction RUL
- `POST /api/v1/rul/predict/batch` - Prédiction batch
- `POST /api/v1/rul/train` - Entraînement des modèles
- `POST /api/v1/rul/calibrate` - Calibration
- `GET /api/v1/rul/status` - Statut du service
- `GET /api/v1/rul/` - Historique des prédictions

**Responsabilités :**
- Gestion des requêtes HTTP
- Validation des données (Pydantic)
- Orchestration des services
- Journalisation PostgreSQL

### 2. Services de Modèles

#### LSTM Service
**Fichier** : `app/services/lstm_service.py`

- Modèle LSTM pour séquences temporelles
- Support transfer learning
- Intégration MLflow

#### GRU Service
**Fichier** : `app/services/gru_service.py`

- Modèle GRU (similaire à LSTM mais plus simple)
- Support transfer learning
- Intégration MLflow

#### TCN Service
**Fichier** : `app/services/tcn_service.py`

- Temporal Convolutional Network
- Convolutions temporelles
- Support transfer learning

#### XGBoost Service
**Fichier** : `app/services/xgboost_service.py`

- Gradient Boosting pour RUL
- Pas de séquences (features statiques)
- Intégration MLflow

### 3. RUL Prediction Service

**Fichier** : `app/services/rul_prediction_service.py`

**Responsabilités :**
- Orchestration de tous les modèles
- Agrégation d'ensemble
- Gestion de la calibration
- Quantification d'incertitude
- Support transfer learning

**Flux de prédiction :**
```
Request → RULPredictionService
    ↓
1. Vérifier modèles entraînés
    ↓
2. Préparer séquences (si nécessaire)
    ↓
3. Prédire avec chaque modèle
    ↓
4. Appliquer calibration (si disponible)
    ↓
5. Agréger prédictions (ensemble)
    ↓
6. Calculer incertitude et intervalle de confiance
    ↓
7. Retourner RULPredictionResult
```

### 4. Calibration Service

**Fichier** : `app/services/calibration_service.py`

**Méthodes :**
- **Isotonic Regression** : Régression isotonique
- **Platt Scaling** : Scaling logistique
- **Temperature Scaling** : Scaling de température (PyTorch)

**Objectif :** Améliorer la calibration des prédictions pour refléter mieux les vraies probabilités/valeurs.

### 5. Transfer Learning Service

**Fichier** : `app/services/transfer_learning_service.py`

**Fonctionnalités :**
- Chargement de modèles pré-entraînés (NASA C-MAPSS)
- Gel de couches (freeze layers)
- Fine-tuning sur données spécifiques

**Workflow :**
```
1. Charger modèle pré-entraîné (.pth)
    ↓
2. Optionnel : Geler certaines couches
    ↓
3. Fine-tuning sur données spécifiques
    ↓
4. Évaluation et déploiement
```

### 6. Kafka Integration

#### Consumer
**Fichier** : `app/services/kafka_consumer.py`

- Consomme les features depuis `extracted-features`
- Désérialisation JSON
- Gestion des erreurs

#### Producer
**Fichier** : `app/services/kafka_producer.py`

- Publie les prédictions RUL sur `rul-predictions`
- Sérialisation JSON
- Callbacks de livraison

#### Worker
**Fichier** : `app/worker.py`

**Flux :**
```
Kafka Consumer → Features
    ↓
RUL Prediction Service → Prédiction
    ↓
Kafka Producer → RUL Predictions
    ↓
PostgreSQL Service → Journalisation
```

### 7. PostgreSQL Service

**Fichier** : `app/database/postgresql.py`

**Table :** `rul_predictions`

**Fonctionnalités :**
- Insertion de prédictions
- Récupération avec filtres
- Pagination
- Comptage
- Dernière prédiction

### 8. MLflow Service

**Fichier** : `app/services/mlflow_service.py`

**Fonctionnalités :**
- Tracking des expériences
- Logging de paramètres et métriques
- Registre des modèles
- Chargement de modèles

## Flux de Données

### 1. Prédiction via API REST

```
Client → FastAPI → RULPredictionService
    ↓
    ├─→ LSTM Service → Prédiction
    ├─→ GRU Service → Prédiction
    ├─→ TCN Service → Prédiction
    └─→ XGBoost Service → Prédiction
    ↓
Calibration Service → Calibration
    ↓
Agrégation Ensemble → RULPredictionResult
    ↓
PostgreSQL Service → Journalisation
    ↓
Client ← Réponse JSON
```

### 2. Prédiction via Kafka Worker

```
Kafka (extracted-features)
    ↓
Kafka Consumer → Features
    ↓
RULPredictionService → Prédiction
    ↓
Kafka Producer → Kafka (rul-predictions)
    ↓
PostgreSQL Service → Journalisation
```

### 3. Entraînement

```
Client → FastAPI → RULPredictionService
    ↓
    ├─→ LSTM Service → Entraînement → MLflow
    ├─→ GRU Service → Entraînement → MLflow
    ├─→ TCN Service → Entraînement → MLflow
    └─→ XGBoost Service → Entraînement → MLflow
    ↓
TrainingResult → Client
```

## Modèles de Données

### RULPredictionRequest
```python
{
    "asset_id": str,
    "sensor_id": Optional[str],
    "features": Dict[str, float],
    "sequence_data": Optional[List[Dict[str, float]]],
    "timestamp": Optional[datetime],
    "metadata": Optional[Dict]
}
```

### RULPredictionResult
```python
{
    "asset_id": str,
    "sensor_id": Optional[str],
    "timestamp": datetime,
    "rul_prediction": float,
    "confidence_interval_lower": float,
    "confidence_interval_upper": float,
    "confidence_level": float,
    "uncertainty": float,
    "model_used": str,
    "model_scores": Dict[str, float],
    "features": Dict[str, float],
    "metadata": Dict
}
```

## Configuration

### Variables d'Environnement

**Service :**
- `SERVICE_NAME` : Nom du service
- `SERVICE_PORT` : Port HTTP
- `LOG_LEVEL` : Niveau de log

**Kafka :**
- `KAFKA_BOOTSTRAP_SERVERS` : Serveurs Kafka
- `KAFKA_CONSUMER_GROUP` : Groupe de consommateurs
- `KAFKA_TOPIC_INPUT_FEATURES` : Topic d'entrée
- `KAFKA_TOPIC_OUTPUT_RUL` : Topic de sortie

**Database :**
- `DATABASE_HOST` : Hôte PostgreSQL
- `DATABASE_PORT` : Port PostgreSQL
- `DATABASE_NAME` : Nom de la base
- `DATABASE_USER` : Utilisateur
- `DATABASE_PASSWORD` : Mot de passe

**MLflow :**
- `MLFLOW_TRACKING_URI` : URI MLflow
- `MLFLOW_EXPERIMENT_NAME` : Nom de l'expérience
- `MLFLOW_ENABLED` : Activer/désactiver MLflow

**Modèles :**
- `ENABLE_LSTM`, `ENABLE_GRU`, `ENABLE_TCN`, `ENABLE_XGBOOST`
- Paramètres spécifiques à chaque modèle (hidden_size, epochs, etc.)

## Dépendances Externes

### Services
- **Kafka** : Message broker pour streaming
- **PostgreSQL** : Base de données pour journalisation
- **MLflow** : Tracking et registry des modèles

### Bibliothèques Python
- **FastAPI** : Framework web
- **PyTorch** : Deep learning (LSTM, GRU, TCN)
- **XGBoost** : Gradient boosting
- **scikit-learn** : Calibration
- **MLflow** : Tracking ML
- **psycopg2** : Client PostgreSQL
- **confluent-kafka** : Client Kafka

## Performance

### Optimisations
- **Pool de connexions** : PostgreSQL (1-10 connexions)
- **Batch processing** : Prédictions batch pour améliorer le débit
- **Caching** : Cache des modèles (si activé)
- **GPU** : Support CUDA pour PyTorch

### Métriques
- **Latence** : < 100ms pour une prédiction simple
- **Débit** : > 1000 prédictions/seconde (batch)
- **Précision** : MAE < 5 cycles (selon données)

## Sécurité

### Recommandations
- **HTTPS** : Utiliser HTTPS en production
- **Authentification** : Ajouter JWT/OAuth2
- **Validation** : Validation stricte des entrées (Pydantic)
- **Rate Limiting** : Limiter le nombre de requêtes
- **Secrets** : Gérer les secrets via variables d'environnement

## Scalabilité

### Horizontal Scaling
- **Stateless** : Le service est stateless (sauf modèles en mémoire)
- **Load Balancer** : Utiliser un load balancer (Nginx, HAProxy)
- **Multiple Instances** : Déployer plusieurs instances

### Vertical Scaling
- **GPU** : Utiliser GPU pour accélérer l'entraînement
- **RAM** : Augmenter RAM pour modèles plus grands
- **CPU** : Plus de CPU pour traitement parallèle

## Monitoring

### Métriques à Surveiller
- **Latence API** : Temps de réponse
- **Taux d'erreur** : Erreurs 4xx/5xx
- **Utilisation CPU/RAM** : Ressources système
- **Taille des modèles** : Taille en mémoire
- **Précision** : MAE, RMSE, R²

### Logs
- **Structured Logging** : JSON logs
- **Niveaux** : DEBUG, INFO, WARNING, ERROR
- **Rotation** : Rotation des logs

## Déploiement

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8085"]
```

### Kubernetes
- **Deployment** : Déploiement avec réplicas
- **Service** : Service ClusterIP/NodePort
- **ConfigMap** : Configuration
- **Secret** : Secrets (DB, Kafka)

## Évolutions Futures

1. **Modèles additionnels** : Transformer, Attention mechanisms
2. **Explicabilité** : SHAP, LIME pour expliquer les prédictions
3. **AutoML** : Sélection automatique de modèles
4. **Federated Learning** : Apprentissage distribué
5. **Edge Computing** : Déploiement sur edge devices

