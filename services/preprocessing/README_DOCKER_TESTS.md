# Guide Docker pour les Tests - Résumé

## 🚀 Utilisation Rapide

### Commandes de Base

```powershell
# Se placer dans le répertoire
cd services/preprocessing

# Exécuter tous les tests
.\scripts\run-tests-docker.ps1

# Tests unitaires uniquement
.\scripts\run-tests-docker.ps1 -TestType unit

# Tests d'intégration uniquement
.\scripts\run-tests-docker.ps1 -TestType integration

# Tests avec couverture
.\scripts\run-tests-docker.ps1 -Coverage
```

---

## 📋 Ce qui est Configuré

### 1. Dockerfile.test
- Image Python 3.11-slim
- Dépendances installées
- Configuration optimisée

### 2. docker-compose.test.yml
- Services : Kafka, Zookeeper, PostgreSQL
- Réseau : predictive-maintenance
- Ports : 9093 (Kafka), 5433 (PostgreSQL), 2181 (Zookeeper)

### 3. Scripts
- `run-tests-docker.ps1` : PowerShell (Windows)
- `run-tests-docker.sh` : Bash (Linux/Mac)

### 4. Documentation
- `QUICK_START_DOCKER_TESTS.md` : Guide rapide
- `DOCKER_TESTING_GUIDE.md` : Guide complet
- `PHASE_3_DOCKER_TESTING_SETUP.md` : Configuration

---

## ✅ Avantages

1. **Isolation complète** : Pas de conflits avec le système hôte
2. **Résolution des problèmes Windows** : Pas d'encodage, pas de timeout
3. **Facilité d'utilisation** : Scripts automatisés
4. **Reproductibilité** : Configuration centralisée

---

## 🐛 Résolution de Problèmes

### Docker non démarré
```powershell
# Démarrer Docker Desktop
# Attendre que Docker soit prêt
# Réexécuter le script
```

### Ports déjà utilisés
```powershell
# Arrêter les services existants
docker-compose -f docker-compose.test.yml down

# Vérifier les ports
netstat -an | findstr "9093 5433 2181"
```

### Réseau manquant
```powershell
# Créer le réseau
docker network create predictive-maintenance
```

---

## 📚 Documentation Complète

- `QUICK_START_DOCKER_TESTS.md` : Guide rapide étape par étape
- `DOCKER_TESTING_GUIDE.md` : Guide complet avec détails
- `PHASE_3_DOCKER_TESTING_SETUP.md` : Configuration et architecture

---

**Prêt à tester ?** Exécutez `.\scripts\run-tests-docker.ps1` maintenant !

