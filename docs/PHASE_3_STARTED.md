# Phase 3 : Service Prétraitement - 🚧 DÉMARRÉE

## Date : 13 novembre 2025

---

## 🎯 Objectifs de la Phase 3

### Fonctionnalités Principales

1. **Consumer Kafka**
   - Consommer depuis topic `sensor-data`
   - Gestion des offsets
   - Gestion d'erreurs et retry

2. **Nettoyage des Données**
   - Détection et suppression des outliers (IQR, Z-score)
   - Gestion des valeurs manquantes
   - Validation qualité des données

3. **Rééchantillonnage**
   - Synchronisation multi-capteurs
   - Rééchantillonnage à fréquence fixe
   - Interpolation (linéaire, spline)

4. **Débruitage**
   - Filtres passe-bande
   - Filtres de Kalman (optionnel)
   - Réduction du bruit

5. **Analyse Fréquentielle**
   - STFT (Short-Time Fourier Transform)
   - FFT (Fast Fourier Transform)
   - Extraction de caractéristiques fréquentielles

6. **Fenêtrage Glissant**
   - Fenêtres de taille fixe pour ML
   - Chevauchement configurable
   - Génération de séquences

7. **Producer Kafka**
   - Publication sur topic `preprocessed-data`
   - Format standardisé
   - Métadonnées de prétraitement

8. **Stockage TimescaleDB**
   - Stockage des données prétraitées
   - Traçabilité des transformations
   - Historique des fenêtres

---

## 📦 Structure Créée

### Fichiers de Base
- ✅ `services/preprocessing/README.md`
- ✅ `services/preprocessing/requirements.txt`
- ✅ `services/preprocessing/app/__init__.py`
- ✅ `services/preprocessing/app/config.py`
- ✅ `services/preprocessing/app/models/sensor_data.py`

### Structure de Dossiers
```
services/preprocessing/
├── app/
│   ├── models/          ✅
│   ├── services/        ⏳ À créer
│   ├── database/        ⏳ À créer
│   └── api/            ⏳ À créer
├── tests/              ⏳ À créer
└── requirements.txt    ✅
```

---

## 🔧 Technologies

- **Python 3.11+**
- **FastAPI** - Framework web
- **Pandas** - Manipulation données
- **SciPy** - Traitement signal
- **NumPy** - Calculs numériques
- **confluent-kafka** - Client Kafka
- **psycopg2** - Client PostgreSQL/TimescaleDB

---

## 📋 Prochaines Étapes

### Étape 1 : Services de Base (En cours)
- [x] Configuration
- [x] Modèles de données
- [ ] Service Kafka Consumer
- [ ] Service Kafka Producer
- [ ] Service de nettoyage

### Étape 2 : Services de Traitement
- [ ] Service de rééchantillonnage
- [ ] Service de débruitage
- [ ] Service d'analyse fréquentielle
- [ ] Service de fenêtrage

### Étape 3 : Intégration
- [ ] Service principal (orchestration)
- [ ] API REST
- [ ] Accès TimescaleDB
- [ ] Tests

### Étape 4 : Tests et Validation
- [ ] Tests unitaires
- [ ] Tests d'intégration
- [ ] Tests avec données NASA C-MAPSS
- [ ] Validation performance

---

## 📊 Progression

**Phase 3** : 🚧 **5% COMPLÉTÉE**

- [x] Structure de base créée
- [x] Configuration
- [x] Modèles de données
- [ ] Services implémentés (0/8)
- [ ] Tests créés (0/5)
- [ ] Documentation complète

---

**Statut** : 🚧 **DÉMARRÉE** - Structure de base en place

**Prochaine Action** : Implémenter les services de base (Kafka Consumer/Producer, Nettoyage)

