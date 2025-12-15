# Phase 6 - Service Detection Anomalies - Démarrage

## Statut

🚧 **Phase 6 en cours de développement**

## Ce qui a été fait

✅ **Structure de base créée** :
- `requirements.txt` avec toutes les dépendances (PyOD, PyTorch, MLflow, FastAPI, Kafka)
- `app/config.py` - Configuration complète du service
- `app/main.py` - Point d'entrée FastAPI avec health check
- `app/models/anomaly_data.py` - Modèles Pydantic pour les données
- `README.md` - Documentation mise à jour

## Prochaines étapes

1. **Implémenter les modèles PyOD** :
   - `app/services/isolation_forest_service.py`
   - `app/services/one_class_svm_service.py`

2. **Implémenter LSTM Autoencoder** :
   - `app/services/lstm_autoencoder_service.py`
   - Architecture : Encoder [64, 32, 16] → Decoder [16, 32, 64]

3. **Service de détection principal** :
   - `app/services/anomaly_detection_service.py` - Orchestration des modèles

4. **Consumer Kafka** :
   - `app/services/kafka_consumer.py` - Consommer le topic "extracted-features"

5. **MLflow Service** :
   - `app/services/mlflow_service.py` - Tracking des expériences

6. **API FastAPI** :
   - `app/api/anomalies.py` - Endpoints REST

7. **Base de données** :
   - `app/database/postgresql.py` - Journalisation des anomalies

8. **Tests** :
   - Tests unitaires pour chaque modèle
   - Tests d'intégration
   - Tests de performance

## Notes

- Phase 5 (KNIME) mise en pause - peut être complétée plus tard
- On utilise directement les features de la Phase 4
- Les modèles seront entraînés sur le dataset NASA C-MAPSS

