# Guide de Test avec Docker - Service Prétraitement

## Date : 13 novembre 2025

---

## 📋 Prérequis

### Docker
- ✅ Docker Desktop installé et démarré
- ✅ docker-compose installé
- ✅ Au moins 4 GB de RAM disponibles
- ✅ Au moins 10 GB d'espace disque disponible

### Réseau Docker
- ✅ Réseau `predictive-maintenance` créé (créé automatiquement)

---

## 🚀 Démarrage Rapide

### Option 1 : Script PowerShell (Windows)

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

### Option 2 : Script Bash (Linux/Mac)

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

### Option 3 : Commandes Docker Manuelles

```bash
# 1. Construire l'image de test
docker build -f Dockerfile.test -t preprocessing-test:latest .

# 2. Créer le réseau
docker network create predictive-maintenance

# 3. Démarrer les services dépendants
docker-compose -f docker-compose.test.yml up -d kafka zookeeper postgresql

# 4. Attendre que les services soient prêts
sleep 30

# 5. Exécuter les tests
docker-compose -f docker-compose.test.yml run --rm preprocessing-test pytest tests/ -v

# 6. Arrêter les services
docker-compose -f docker-compose.test.yml down
```

---

## 📊 Types de Tests

### Tests Unitaires

```powershell
# PowerShell
.\scripts\run-tests-docker.ps1 -TestType unit

# Bash
./scripts/run-tests-docker.sh unit

# Docker
docker-compose -f docker-compose.test.yml run --rm preprocessing-test pytest tests/ -v -m "not integration"
```

**Tests inclus** :
- ✅ CleaningService (6 tests)
- ✅ ResamplingService (3 tests)
- ✅ DenoisingService (4 tests)
- ✅ FrequencyAnalysisService (4 tests)
- ✅ WindowingService (4 tests)
- ✅ PreprocessingService (5 tests)

### Tests d'Intégration

```powershell
# PowerShell
.\scripts\run-tests-docker.ps1 -TestType integration

# Bash
./scripts/run-tests-docker.sh integration

# Docker
docker-compose -f docker-compose.test.yml run --rm preprocessing-test pytest tests/ -v -m integration
```

**Tests inclus** :
- ✅ Tests Kafka (4 tests)
- ✅ Tests TimescaleDB (5 tests)
- ✅ Tests End-to-End (2 tests)

### Tous les Tests

```powershell
# PowerShell
.\scripts\run-tests-docker.ps1 -TestType all

# Bash
./scripts/run-tests-docker.sh all

# Docker
docker-compose -f docker-compose.test.yml run --rm preprocessing-test pytest tests/ -v
```

---

## 📈 Couverture de Code

### Avec Couverture

```powershell
# PowerShell
.\scripts\run-tests-docker.ps1 -Coverage

# Bash
./scripts/run-tests-docker.sh all true

# Docker
docker-compose -f docker-compose.test.yml run --rm preprocessing-test pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing
```

### Résultats

- 📊 Rapport HTML : `htmlcov/index.html`
- 📊 Rapport terminal : Affiché dans la console
- 📊 Couverture cible : > 80%

---

## 🔧 Configuration

### Variables d'Environnement

Les variables d'environnement sont configurées dans `docker-compose.test.yml` :

```yaml
environment:
  - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
  - DATABASE_HOST=postgresql
  - DATABASE_PORT=5432
  - DATABASE_NAME=predictive_maintenance
  - DATABASE_USER=pmuser
  - DATABASE_PASSWORD=pmpassword
```

### Modifier la Configuration

1. Modifier `docker-compose.test.yml`
2. Reconstruire l'image : `docker build -f Dockerfile.test -t preprocessing-test:latest .`
3. Redémarrer les services : `docker-compose -f docker-compose.test.yml up -d`

---

## 🐛 Dépannage

### Problème 1 : Services non prêts

**Symptôme** : Timeout lors de la vérification des services

**Solution** :
```powershell
# Vérifier manuellement
docker exec kafka-test nc -z localhost 9092
docker exec postgresql-test pg_isready -U pmuser -d predictive_maintenance

# Augmenter le délai d'attente dans le script
# Modifier la ligne : Start-Sleep -Seconds 30
```

### Problème 2 : Réseau Docker manquant

**Symptôme** : Erreur "network not found"

**Solution** :
```powershell
# Créer le réseau
docker network create predictive-maintenance

# Vérifier que le réseau existe
docker network ls | grep predictive-maintenance
```

### Problème 3 : Ports déjà utilisés

**Symptôme** : Erreur "port is already allocated"

**Solution** :
```powershell
# Arrêter les services existants
docker-compose -f docker-compose.test.yml down

# Vérifier les ports
netstat -an | findstr "9092 5432 2181"

# Modifier les ports dans docker-compose.test.yml si nécessaire
```

### Problème 4 : Image non construite

**Symptôme** : Erreur "image not found"

**Solution** :
```powershell
# Construire l'image
docker build -f Dockerfile.test -t preprocessing-test:latest .

# Vérifier que l'image existe
docker images | grep preprocessing-test
```

---

## 📋 Checklist de Test

### Avant de Commencer

- [ ] Docker Desktop démarré
- [ ] docker-compose installé
- [ ] Réseau Docker créé
- [ ] Image de test construite
- [ ] Services dépendants démarrés

### Exécution des Tests

- [ ] Tests unitaires exécutés
- [ ] Tests d'intégration exécutés
- [ ] Tous les tests passent
- [ ] Couverture > 80%
- [ ] Rapport de couverture généré

### Après les Tests

- [ ] Services arrêtés (optionnel)
- [ ] Rapport de couverture consulté
- [ ] Résultats validés
- [ ] Problèmes identifiés documentés

---

## 🎯 Résultats Attendus

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

## 📚 Ressources

### Documentation

- [Docker Documentation](https://docs.docker.com/)
- [docker-compose Documentation](https://docs.docker.com/compose/)
- [pytest Documentation](https://docs.pytest.org/)
- [TimescaleDB Documentation](https://docs.timescale.com/)

### Scripts

- `scripts/run-tests-docker.ps1` : Script PowerShell
- `scripts/run-tests-docker.sh` : Script Bash
- `Dockerfile.test` : Dockerfile pour les tests
- `docker-compose.test.yml` : Configuration Docker Compose

---

## ✅ Avantages de Docker pour les Tests

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

**Statut** : ✅ **Guide de test Docker créé**

**Recommandation** : Utiliser Docker pour tous les tests d'intégration pour éviter les problèmes d'encodage et de configuration Windows

