# Guide : Comment Exécuter les Tests

## Date : 13 novembre 2025

---

## ✅ Solution Recommandée : Tests Locaux

### Pourquoi

1. ✅ **Pas de problème de proxy Docker** : Utilise Python localement
2. ✅ **Plus rapide** : Pas besoin de construire une image Docker
3. ✅ **Plus simple** : Utilise l'infrastructure Docker existante
4. ✅ **Même résultats** : Tests identiques

---

## 🚀 Utilisation Rapide

### Étape 1 : Démarrer l'Infrastructure (si nécessaire)

```powershell
# Vérifier que l'infrastructure est démarrée
docker ps | findstr "kafka postgresql"

# Si l'infrastructure n'est pas démarrée
cd infrastructure
docker-compose up -d
```

### Étape 2 : Exécuter les Tests Locaux

```powershell
# Se placer dans le répertoire du service
cd services/preprocessing

# Exécuter tous les tests
.\scripts\run-tests-local.ps1

# Tests unitaires uniquement
.\scripts\run-tests-local.ps1 -TestType unit

# Tests d'intégration uniquement
.\scripts\run-tests-local.ps1 -TestType integration

# Tests avec couverture
.\scripts\run-tests-local.ps1 -Coverage
```

### Étape 3 : Consulter les Résultats

```powershell
# Résultats dans le terminal
# Rapport de couverture : htmlcov/index.html
start htmlcov/index.html
```

---

## 📊 Résultats Attendus

### Tests Unitaires

```
tests/test_cleaning_service.py::TestCleaningService::test_clean_single_value_good_quality PASSED
tests/test_cleaning_service.py::TestCleaningService::test_clean_single_value_bad_quality PASSED
...
============================= 26 passed in 16.00s =============================
```

### Tests d'Intégration

```
tests/test_integration_kafka.py::TestKafkaIntegration::test_kafka_producer_connection PASSED
tests/test_integration_timescaledb.py::TestTimescaleDBIntegration::test_timescaledb_connection PASSED
...
============================= 11 passed in 45.23s =============================
```

### Couverture

```
---------- coverage: platform win32, Python 3.12.6 -----------
Name                                         Stmts   Miss  Cover
-----------------------------------------------------------------
app/services/cleaning_service.py               80     11    86%
app/services/denoising_service.py               76     21    72%
...
-----------------------------------------------------------------
TOTAL                                          901    352    61%
```

---

## 🔧 Options Disponibles

### Tests Unitaires Seulement

```powershell
.\scripts\run-tests-local.ps1 -TestType unit
```

### Tests d'Intégration Seulement

```powershell
# Nécessite Kafka et PostgreSQL démarrés
.\scripts\run-tests-local.ps1 -TestType integration
```

### Tous les Tests

```powershell
.\scripts\run-tests-local.ps1 -TestType all
```

### Tests avec Couverture

```powershell
.\scripts\run-tests-local.ps1 -Coverage
```

---

## 🐛 Dépannage

### Problème : Infrastructure non démarrée

**Solution** :
```powershell
# Démarrer l'infrastructure
cd infrastructure
docker-compose up -d

# Vérifier que les services sont démarrés
docker ps | findstr "kafka postgresql"
```

### Problème : Tests d'intégration échouent

**Solution** :
```powershell
# Vérifier que Kafka est démarré
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Vérifier que PostgreSQL est démarré
docker exec postgresql psql -U pmuser -d predictive_maintenance -c "SELECT 1;"
```

### Problème : Problème de proxy Docker

**Solution** :
```powershell
# Utiliser les tests locaux (pas besoin de Docker Hub)
.\scripts\run-tests-local.ps1
```

---

## ✅ Avantages des Tests Locaux

1. ✅ **Pas de problème de proxy** : Utilise Python localement
2. ✅ **Plus rapide** : Pas besoin de construire une image Docker
3. ✅ **Plus simple** : Utilise l'infrastructure Docker existante
4. ✅ **Même résultats** : Tests identiques

---

## 📚 Documentation Complète

- `HOW_TO_RUN_TESTS.md` : Ce guide
- `SOLUTION_PROXY_DOCKER.md` : Solution pour problème de proxy
- `FIX_DOCKER_PROXY_COMPLETE.md` : Guide complet
- `DOCKER_TESTING_GUIDE.md` : Guide Docker

---

## 🎯 Résumé

### Solution Recommandée

```powershell
# 1. Démarrer l'infrastructure
cd infrastructure
docker-compose up -d

# 2. Exécuter les tests
cd ../services/preprocessing
.\scripts\run-tests-local.ps1
```

### Résultats

- ✅ **26/26 tests unitaires passent (100%)**
- ✅ **Couverture : 61%**
- ✅ **Aucune erreur critique**

---

**Prêt à tester ?** Exécutez `.\scripts\run-tests-local.ps1` maintenant !

