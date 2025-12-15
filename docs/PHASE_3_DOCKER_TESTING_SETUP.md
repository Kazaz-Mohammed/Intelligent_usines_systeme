# Configuration des Tests Docker - Service Prétraitement

## Date : 13 novembre 2025

---

## ✅ Configuration Créée

### Fichiers Créés

1. **Dockerfile.test** ✅
   - Image Docker pour les tests
   - Python 3.11-slim
   - Dépendances installées
   - Configuration optimisée

2. **docker-compose.test.yml** ✅
   - Configuration Docker Compose pour les tests
   - Services : Kafka, Zookeeper, PostgreSQL
   - Réseau : predictive-maintenance
   - Volumes : données de test

3. **scripts/run-tests-docker.ps1** ✅
   - Script PowerShell pour Windows
   - Automatisation complète
   - Gestion des services
   - Rapports de couverture

4. **scripts/run-tests-docker.sh** ✅
   - Script Bash pour Linux/Mac
   - Automatisation complète
   - Gestion des services
   - Rapports de couverture

5. **DOCKER_TESTING_GUIDE.md** ✅
   - Guide complet de test
   - Instructions détaillées
   - Dépannage
   - Exemples d'utilisation

6. **.dockerignore.test** ✅
   - Fichiers à exclure de l'image
   - Optimisation de la taille
   - Performance améliorée

---

## 🚀 Utilisation

### Démarrage Rapide

```powershell
# Windows PowerShell
cd services/preprocessing
.\scripts\run-tests-docker.ps1

# Linux/Mac Bash
cd services/preprocessing
./scripts/run-tests-docker.sh
```

### Options Disponibles

#### PowerShell

```powershell
# Tous les tests
.\scripts\run-tests-docker.ps1

# Tests unitaires uniquement
.\scripts\run-tests-docker.ps1 -TestType unit

# Tests d'intégration uniquement
.\scripts\run-tests-docker.ps1 -TestType integration

# Tests avec couverture
.\scripts\run-tests-docker.ps1 -Coverage

# Garder les services en cours d'exécution
.\scripts\run-tests-docker.ps1 -KeepServices
```

#### Bash

```bash
# Tous les tests
./scripts/run-tests-docker.sh

# Tests unitaires uniquement
./scripts/run-tests-docker.sh unit

# Tests d'intégration uniquement
./scripts/run-tests-docker.sh integration

# Tests avec couverture
./scripts/run-tests-docker.sh all true
```

---

## 🔧 Architecture

### Services Docker

1. **preprocessing-test** (service de test)
   - Image : preprocessing-test:latest
   - Réseau : predictive-maintenance
   - Dépendances : Kafka, PostgreSQL

2. **kafka** (service Kafka)
   - Image : confluentinc/cp-kafka:7.5.0
   - Port : 9092
   - Health check : nc -z localhost 9092

3. **zookeeper** (service Zookeeper)
   - Image : confluentinc/cp-zookeeper:7.5.0
   - Port : 2181
   - Health check : nc -z localhost 2181

4. **postgresql** (service PostgreSQL + TimescaleDB)
   - Image : timescale/timescaledb:latest-pg16
   - Port : 5432
   - Health check : pg_isready

### Réseau Docker

- **Réseau** : predictive-maintenance
- **Type** : bridge
- **Services** : Tous les services de test

### Volumes Docker

- **postgresql-test-data** : Données PostgreSQL
- **htmlcov** : Rapports de couverture

---

## 📊 Résultats Attendus

### Tests Unitaires

- ✅ **28/28 tests passent (100%)**
- ✅ Couverture : > 80%
- ✅ Aucune erreur critique

### Tests d'Intégration

- ✅ **11/11 tests passent (100%)**
- ✅ Kafka fonctionne
- ✅ TimescaleDB fonctionne
- ✅ Pipeline end-to-end fonctionne

### Couverture Globale

- ✅ **Couverture : > 80%**
- ✅ Services principaux : > 90%
- ✅ Services d'intégration : > 70%

---

## 🐛 Dépannage

### Problème 1 : Services non prêts

**Solution** :
```powershell
# Vérifier manuellement
docker exec kafka-test nc -z localhost 9092
docker exec postgresql-test pg_isready -U pmuser -d predictive_maintenance

# Augmenter le délai d'attente
# Modifier dans le script : Start-Sleep -Seconds 30
```

### Problème 2 : Réseau Docker manquant

**Solution** :
```powershell
# Créer le réseau
docker network create predictive-maintenance

# Vérifier
docker network ls | grep predictive-maintenance
```

### Problème 3 : Ports déjà utilisés

**Solution** :
```powershell
# Arrêter les services existants
docker-compose -f docker-compose.test.yml down

# Vérifier les ports
netstat -an | findstr "9092 5432 2181"
```

### Problème 4 : Image non construite

**Solution** :
```powershell
# Construire l'image
docker build -f Dockerfile.test -t preprocessing-test:latest .

# Vérifier
docker images | grep preprocessing-test
```

---

## ✅ Avantages

### 1. Isolation Complète

- ✅ Environnement isolé
- ✅ Pas de conflits avec le système hôte
- ✅ Configuration reproductible

### 2. Résolution des Problèmes Windows

- ✅ Pas de problème d'encodage UTF-8
- ✅ Pas de problème de timeout Kafka
- ✅ Configuration réseau simplifiée

### 3. Facilité d'Utilisation

- ✅ Scripts automatisés
- ✅ Configuration centralisée
- ✅ Résultats reproductibles

### 4. Intégration CI/CD

- ✅ Prêt pour l'intégration continue
- ✅ Tests automatisés
- ✅ Rapports de couverture

---

## 📚 Documentation

### Guides

- ✅ `DOCKER_TESTING_GUIDE.md` : Guide complet
- ✅ `PHASE_3_DOCKER_TESTING_SETUP.md` : Configuration
- ✅ `PHASE_3_INTEGRATION_TEST_RESULTS.md` : Résultats

### Scripts

- ✅ `scripts/run-tests-docker.ps1` : Script PowerShell
- ✅ `scripts/run-tests-docker.sh` : Script Bash
- ✅ `Dockerfile.test` : Dockerfile pour les tests
- ✅ `docker-compose.test.yml` : Configuration Docker Compose

---

## 🎯 Prochaines Étapes

### 1. Exécuter les Tests

```powershell
# Exécuter tous les tests
.\scripts\run-tests-docker.ps1

# Vérifier les résultats
# Consulter htmlcov/index.html pour la couverture
```

### 2. Valider les Résultats

- ✅ Tous les tests passent
- ✅ Couverture > 80%
- ✅ Aucune erreur critique

### 3. Intégrer dans CI/CD

- ✅ Configurer GitHub Actions
- ✅ Configurer GitLab CI
- ✅ Configurer Jenkins

---

**Statut** : ✅ **Configuration Docker complète pour les tests**

**Recommandation** : Utiliser Docker pour tous les tests d'intégration pour éviter les problèmes d'encodage et de configuration Windows

