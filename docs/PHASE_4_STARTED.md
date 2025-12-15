# Phase 4 : Service Extraction Features - DÉMARRÉE

## Date : 13 novembre 2025

---

## ✅ Phase 4 Démarrée

### Branche Créée

- **Branche** : `feature/service-extraction-features`
- **Basé sur** : `develop`
- **Date** : 13 novembre 2025

---

## 📋 Objectifs de la Phase 4

### Service Extraction Features

**Rôle** :
- Calcul de caractéristiques temporelles/fréquentielles depuis les données prétraitées
- Intégration Feature Store (Feast)
- Standardisation par type d'actif
- Publication des features sur Kafka

**Inputs** :
- Topic Kafka : `preprocessed-data` (données prétraitées)
- Topic Kafka : `windowed-data` (fenêtres pour ML)
- TimescaleDB : tables `preprocessed_sensor_data`, `windowed_sensor_data`

**Outputs** :
- Topic Kafka : `extracted-features` (features calculées)
- TimescaleDB : table `extracted_features`
- Feast Feature Store : features stockées et versionnées

**Technologies** :
- Python/FastAPI
- tsfresh ou tsflex pour features temporelles
- Feast pour feature store
- SciPy pour features fréquentielles
- XGBoost pour standardisation (optionnel)

---

## 📊 Plan de Développement

### Étape 1 : Structure et Configuration

1. ✅ Créer branche `feature/service-extraction-features`
2. ⏳ Créer structure Python/FastAPI
3. ⏳ Configuration FastAPI et dépendances
4. ⏳ Modèles de données (ExtractedFeatures)

### Étape 2 : Services de Base

1. ⏳ Consumer Kafka (topic `preprocessed-data`)
2. ⏳ Producer Kafka (topic `extracted-features`)
3. ⏳ Service TimescaleDB

### Étape 3 : Calcul de Features

1. ⏳ Service calcul features temporelles (tsfresh/tsflex)
2. ⏳ Service calcul features fréquentielles (SciPy)
3. ⏳ Service agrégation features

### Étape 4 : Feature Store

1. ⏳ Configuration Feast
2. ⏳ Service intégration Feast
3. ⏳ Stockage et versioning

### Étape 5 : Standardisation

1. ⏳ Service standardisation par type d'actif
2. ⏳ Normalisation des features

### Étape 6 : Orchestration

1. ⏳ Service principal (orchestration)
2. ⏳ Worker en arrière-plan
3. ⏳ API REST

### Étape 7 : Tests

1. ⏳ Tests unitaires
2. ⏳ Tests d'intégration
3. ⏳ Tests avec Feast

### Étape 8 : Dockerfile

1. ⏳ Dockerfile
2. ⏳ Configuration Docker Compose
3. ⏳ Health checks

---

## 🎯 Prochaines Étapes Immédiates

1. **Créer structure Python/FastAPI**
   - Répertoire `services/extraction-features/`
   - Structure de dossiers
   - Fichiers de base

2. **Configuration**
   - `requirements.txt` avec dépendances
   - `app/config.py` avec configuration
   - Modèles de données

3. **Services de base**
   - Kafka Consumer/Producer
   - TimescaleDB Service

---

## 📚 Documentation Référence

- `DEVELOPMENT_PLAN.md` : Plan complet de développement
- `services/preprocessing/` : Référence pour structure Python/FastAPI
- `CURRENT_STATUS.md` : État actuel du projet

---

**Phase 4 : 🚀 DÉMARRÉE**

**Prochaine Étape** : Créer structure Python/FastAPI

