# Guide de Test - Service Prétraitement

## Date : 13 novembre 2025

---

## 📋 Prérequis

### Infrastructure Docker
- ✅ Docker Desktop doit être en cours d'exécution
- ✅ Infrastructure Docker (Kafka, PostgreSQL, etc.) doit être démarrée

### Python
- ✅ Python 3.11+ installé
- ✅ pip installé
- ✅ Environnement virtuel (optionnel mais recommandé)

---

## 🧪 Tests Unitaires

### 1. Installation des Dépendances

```bash
cd services/preprocessing
pip install -r requirements.txt
```

### 2. Exécution des Tests

```bash
# Tous les tests
pytest

# Tests unitaires uniquement
pytest -m unit

# Tests d'intégration uniquement
pytest -m integration

# Avec couverture
pytest --cov=app --cov-report=html

# Tests spécifiques
pytest tests/test_cleaning_service.py -v
pytest tests/test_resampling_service.py -v
pytest tests/test_denoising_service.py -v
pytest tests/test_frequency_analysis_service.py -v
pytest tests/test_windowing_service.py -v
pytest tests/test_preprocessing_service.py -v
pytest tests/test_integration.py -v
```

### 3. Résultats Attendus

- ✅ Tous les tests doivent passer
- ✅ Couverture > 80%
- ✅ Aucune erreur de syntaxe
- ✅ Aucune erreur d'import

---

## 🚀 Test du Service

### 1. Démarrer l'Infrastructure

```bash
# Démarrer l'infrastructure Docker
cd infrastructure
docker-compose up -d

# Vérifier que les services sont démarrés
docker ps
```

### 2. Vérifier les Tables TimescaleDB

```bash
# Vérifier que les tables existent
docker exec -it postgresql psql -U pmuser -d predictive_maintenance -c "\dt"
```

### 3. Démarrer le Service Prétraitement

```bash
# Option 1 : Directement avec uvicorn
cd services/preprocessing
uvicorn app.main:app --host 0.0.0.0 --port 8082 --reload

# Option 2 : Avec Python
python -m app.main

# Option 3 : Avec Docker
docker-compose -f services/preprocessing/docker-compose.yml up
```

### 4. Vérifier le Health Check

```bash
# Test health endpoint
curl http://localhost:8082/health

# Réponse attendue
{
  "status": "UP",
  "service": "preprocessing-service"
}
```

### 5. Tester les Endpoints REST

```bash
# Root endpoint
curl http://localhost:8082/

# Status endpoint
curl http://localhost:8082/api/v1/preprocessing/status

# Metrics endpoint
curl http://localhost:8082/api/v1/preprocessing/metrics
```

---

## 🔌 Test d'Intégration avec Kafka

### 1. Vérifier les Topics Kafka

```bash
# Lister les topics
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092

# Vérifier que les topics existent
# - sensor-data (input)
# - preprocessed-data (output)
```

### 2. Envoyer des Données de Test

```bash
# Créer un script de test pour envoyer des données
python scripts/test_send_sensor_data.py
```

### 3. Vérifier la Consommation

```bash
# Consommer depuis le topic preprocessed-data
docker exec -it kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic preprocessed-data \
  --from-beginning
```

---

## 💾 Test d'Intégration avec TimescaleDB

### 1. Vérifier l'Insertion de Données

```bash
# Vérifier les données prétraitées
docker exec -it postgresql psql -U pmuser -d predictive_maintenance -c \
  "SELECT COUNT(*) FROM preprocessed_sensor_data;"

# Vérifier les fenêtres
docker exec -it postgresql psql -U pmuser -d predictive_maintenance -c \
  "SELECT COUNT(*) FROM windowed_sensor_data;"
```

### 2. Vérifier les Données Insérées

```bash
# Afficher les dernières données prétraitées
docker exec -it postgresql psql -U pmuser -d predictive_maintenance -c \
  "SELECT * FROM preprocessed_sensor_data ORDER BY time DESC LIMIT 10;"

# Afficher les dernières fenêtres
docker exec -it postgresql psql -U pmuser -d predictive_maintenance -c \
  "SELECT * FROM windowed_sensor_data ORDER BY start_time DESC LIMIT 10;"
```

---

## 📊 Test de Performance

### 1. Test de Charge

```bash
# Envoyer un grand nombre de messages
python scripts/test_load_sensor_data.py --count 1000
```

### 2. Monitoring

```bash
# Vérifier les métriques du service
curl http://localhost:8082/api/v1/preprocessing/metrics

# Vérifier les logs
docker logs preprocessing-service
```

---

## 🐛 Dépannage

### Problèmes Courants

1. **Kafka non accessible**
   - Vérifier que Kafka est démarré : `docker ps | grep kafka`
   - Vérifier les variables d'environnement

2. **PostgreSQL non accessible**
   - Vérifier que PostgreSQL est démarré : `docker ps | grep postgresql`
   - Vérifier les credentials

3. **Tests échouent**
   - Vérifier que les dépendances sont installées
   - Vérifier que Python 3.11+ est utilisé
   - Vérifier les imports

4. **Service ne démarre pas**
   - Vérifier les logs : `docker logs preprocessing-service`
   - Vérifier les variables d'environnement
   - Vérifier les ports

---

## ✅ Checklist de Test

- [ ] Tests unitaires passent
- [ ] Tests d'intégration passent
- [ ] Service démarre correctement
- [ ] Health check fonctionne
- [ ] Endpoints REST fonctionnent
- [ ] Intégration Kafka fonctionne
- [ ] Intégration TimescaleDB fonctionne
- [ ] Données sont correctement prétraitées
- [ ] Fenêtres sont correctement créées
- [ ] Performance acceptable

---

**Statut** : 📋 Guide de test créé

