# Guide Rapide : Tests avec Docker

## 🚀 Démarrage en 3 Étapes

### Étape 1 : Vérifier Docker

```powershell
# Vérifier que Docker est démarré
docker ps

# Vérifier que docker-compose est disponible
docker-compose --version
```

### Étape 2 : Exécuter les Tests

```powershell
# Se placer dans le répertoire du service
cd services/preprocessing

# Exécuter tous les tests
.\scripts\run-tests-docker.ps1
```

### Étape 3 : Consulter les Résultats

```powershell
# Ouvrir le rapport de couverture (si généré)
start htmlcov/index.html
```

---

## 📋 Options Disponibles

### Tests Unitaires Seulement

```powershell
.\scripts\run-tests-docker.ps1 -TestType unit
```

### Tests d'Intégration Seulement

```powershell
.\scripts\run-tests-docker.ps1 -TestType integration
```

### Tests avec Couverture

```powershell
.\scripts\run-tests-docker.ps1 -Coverage
```

### Garder les Services Actifs

```powershell
.\scripts\run-tests-docker.ps1 -KeepServices
```

---

## 🔍 Ce qui se passe en Arrière-Plan

### 1. Construction de l'Image

```powershell
docker build -f Dockerfile.test -t preprocessing-test:latest .
```

**Résultat** : Image Docker `preprocessing-test:latest` créée

### 2. Création du Réseau

```powershell
docker network create predictive-maintenance
```

**Résultat** : Réseau Docker `predictive-maintenance` créé

### 3. Démarrage des Services

```powershell
docker-compose -f docker-compose.test.yml up -d kafka zookeeper postgresql
```

**Résultat** : Services Kafka, Zookeeper et PostgreSQL démarrés

### 4. Vérification des Services

```powershell
# Vérifier Kafka
docker exec kafka-test nc -z localhost 9092

# Vérifier PostgreSQL
docker exec postgresql-test pg_isready -U pmuser -d predictive_maintenance
```

**Résultat** : Services prêts pour les tests

### 5. Exécution des Tests

```powershell
docker-compose -f docker-compose.test.yml run --rm preprocessing-test pytest tests/ -v
```

**Résultat** : Tests exécutés dans un conteneur isolé

---

## 🎯 Exemple Complet

### Scénario : Tous les Tests avec Couverture

```powershell
# 1. Se placer dans le répertoire
cd services/preprocessing

# 2. Exécuter les tests
.\scripts\run-tests-docker.ps1 -TestType all -Coverage

# 3. Attendre la fin des tests (2-5 minutes)
# Les services seront démarrés automatiquement
# Les tests seront exécutés
# Les résultats seront affichés

# 4. Consulter les résultats
# - Terminal : Résultats des tests
# - htmlcov/index.html : Rapport de couverture
```

### Résultat Attendu

```
=== Exécution des tests avec Docker ===

[INFO] Docker est disponible
[INFO] docker-compose est disponible
[INFO] Type de test: all
[INFO] Couverture: True
[INFO] Construction de l'image de test...
[INFO] Image construite avec succès
[INFO] Création du réseau Docker...
[INFO] Démarrage des services dépendants...
[INFO] Attente que les services soient prêts...
[INFO] Vérification de Kafka...
[INFO] Kafka est prêt
[INFO] Vérification de PostgreSQL...
[INFO] PostgreSQL est prêt
[INFO] Exécution des tests...

============================= test session starts =============================
platform linux -- Python 3.11.x, pytest-7.4.3
collected 39 items

tests/test_cleaning_service.py::TestCleaningService::test_clean_single_value_good_quality PASSED
...
tests/test_integration_timescaledb.py::TestTimescaleDBIntegration::test_timescaledb_connection PASSED
...

============================= 39 passed in 45.23s =============================

---------- coverage: platform linux, Python 3.11.x -----------
Name                                         Stmts   Miss  Cover
-----------------------------------------------------------------
app/services/cleaning_service.py               80     11    86%
app/services/denoising_service.py               76     21    72%
...
-----------------------------------------------------------------
TOTAL                                          896    150    83%

[INFO] Tests réussis!
Arrêter les services? (y/n)
```

---

## 🐛 Résolution de Problèmes

### Problème : Docker non démarré

**Symptôme** :
```
error during connect: Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.48/containers/json": open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified
```

**Solution** :
1. Démarrer Docker Desktop
2. Attendre que Docker soit prêt
3. Réexécuter le script

### Problème : Ports déjà utilisés

**Symptôme** :
```
Error response from daemon: Bind for 0.0.0.0:9092 failed: port is already allocated
```

**Solution** :
1. Arrêter les services existants :
   ```powershell
   docker-compose -f docker-compose.test.yml down
   ```
2. Vérifier les ports :
   ```powershell
   netstat -an | findstr "9092 5432 2181"
   ```
3. Arrêter les services utilisant ces ports
4. Réexécuter le script

### Problème : Réseau manquant

**Symptôme** :
```
Error response from daemon: network predictive-maintenance not found
```

**Solution** :
1. Créer le réseau :
   ```powershell
   docker network create predictive-maintenance
   ```
2. Réexécuter le script

### Problème : Services non prêts

**Symptôme** :
```
[ERROR] Kafka n'est pas prêt après 60 secondes
```

**Solution** :
1. Vérifier manuellement :
   ```powershell
   docker exec kafka-test nc -z localhost 9092
   ```
2. Augmenter le délai d'attente dans le script
3. Vérifier les logs :
   ```powershell
   docker logs kafka-test
   ```

---

## ✅ Checklist

### Avant d'Exécuter

- [ ] Docker Desktop démarré
- [ ] docker-compose installé
- [ ] Réseau Docker créé (automatique)
- [ ] Ports disponibles (9092, 5432, 2181)
- [ ] Au moins 4 GB de RAM disponibles

### Après l'Exécution

- [ ] Tous les tests passent
- [ ] Couverture > 80%
- [ ] Rapport de couverture généré
- [ ] Services arrêtés (optionnel)
- [ ] Résultats validés

---

## 📚 Documentation Complète

Pour plus de détails, consultez :
- `DOCKER_TESTING_GUIDE.md` : Guide complet
- `PHASE_3_DOCKER_TESTING_SETUP.md` : Configuration
- `PHASE_3_INTEGRATION_TEST_RESULTS.md` : Résultats

---

## 🎯 Résumé

### Avantages

1. ✅ **Isolation complète** : Pas de conflits avec le système hôte
2. ✅ **Résolution des problèmes Windows** : Pas d'encodage, pas de timeout
3. ✅ **Facilité d'utilisation** : Scripts automatisés
4. ✅ **Reproductibilité** : Configuration centralisée

### Utilisation

```powershell
# Simple
.\scripts\run-tests-docker.ps1

# Avec options
.\scripts\run-tests-docker.ps1 -TestType all -Coverage -KeepServices
```

---

**Prêt à tester ?** Exécutez `.\scripts\run-tests-docker.ps1` maintenant !

