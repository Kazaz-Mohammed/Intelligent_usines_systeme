# Solution Complète : Problème de Proxy Docker

## 🔴 Problème Identifié

Docker essaie toujours d'utiliser un proxy (`http.docker.internal:3128`) même après désactivation :
```
proxyconnect tcp: dial tcp: lookup http.docker.internal on 192.168.65.7:53: i/o timeout
```

## ✅ Solution : Utiliser l'Infrastructure Existante + Tests Locaux

### Pourquoi Cette Solution

1. **Pas de problème de proxy** : Utilise l'infrastructure Docker existante
2. **Pas besoin de Docker Hub** : Utilise Python localement
3. **Plus rapide** : Pas besoin de construire une image Docker
4. **Plus simple** : Utilise ce qui fonctionne déjà

---

## 🚀 Solution Recommandée : Tests Locaux avec Infrastructure Existante

### Étape 1 : Démarrer l'Infrastructure

```powershell
# Se placer dans le répertoire infrastructure
cd infrastructure

# Démarrer l'infrastructure
docker-compose up -d

# Vérifier que les services sont démarrés
docker ps | findstr "kafka postgresql"
```

### Étape 2 : Vérifier que l'Infrastructure est Prête

```powershell
# Vérifier Kafka
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Vérifier PostgreSQL
docker exec postgresql psql -U pmuser -d predictive_maintenance -c "SELECT 1;"
```

### Étape 3 : Exécuter les Tests Locaux

```powershell
# Se placer dans le répertoire du service
cd ../services/preprocessing

# Exécuter les tests localement
.\scripts\run-tests-local.ps1

# Ou avec options
.\scripts\run-tests-local.ps1 -TestType all -Coverage
```

### Étape 4 : Consulter les Résultats

```powershell
# Résultats dans le terminal
# Rapport de couverture : htmlcov/index.html
start htmlcov/index.html
```

---

## 🔧 Correction du Problème de Proxy Docker

### Méthode 1 : Vérifier Docker Engine Configuration

1. **Ouvrir Docker Desktop**
2. **Settings > Docker Engine**
3. **Vérifier la configuration** :
   ```json
   {
     "proxies": {
       "http-proxy": "",
       "https-proxy": "",
       "no-proxy": ""
     }
   }
   ```
4. **Si le proxy est toujours configuré, le supprimer**
5. **Appliquer les changements**
6. **Redémarrer Docker Desktop**

### Méthode 2 : Réinitialiser Docker Desktop

1. **Ouvrir Docker Desktop**
2. **Troubleshoot > Clean / Purge data**
3. **Réinitialiser Docker Desktop**
4. **Redémarrer Docker Desktop**

### Méthode 3 : Vérifier les Variables d'Environnement

```powershell
# Vérifier les variables d'environnement Docker
docker info | Select-String -Pattern "Proxy"

# Vérifier les variables d'environnement système
$env:HTTP_PROXY
$env:HTTPS_PROXY
$env:NO_PROXY
```

---

## 📋 Solution Alternative : Tests Sans Docker

### Option 1 : Tests Locaux Complets

```powershell
# 1. Démarrer l'infrastructure
cd infrastructure
docker-compose up -d

# 2. Exécuter les tests localement
cd ../services/preprocessing
.\scripts\run-tests-local.ps1 -TestType all -Coverage
```

### Option 2 : Tests Unitaires Seulement (Sans Infrastructure)

```powershell
# Tests unitaires uniquement (pas besoin d'infrastructure)
cd services/preprocessing
.\scripts\run-tests-local.ps1 -TestType unit
```

### Option 3 : Utiliser l'Infrastructure Existante avec Docker

```powershell
# 1. Démarrer l'infrastructure
cd infrastructure
docker-compose up -d

# 2. Utiliser l'infrastructure existante pour les tests
cd ../services/preprocessing
.\scripts\run-tests-with-existing-infra.ps1 -BuildImage
```

---

## 🎯 Solution Rapide (Recommandée)

### Étape 1 : Démarrer l'Infrastructure

```powershell
cd infrastructure
docker-compose up -d
```

### Étape 2 : Exécuter les Tests Locaux

```powershell
cd ../services/preprocessing
.\scripts\run-tests-local.ps1
```

### Étape 3 : Consulter les Résultats

- **Terminal** : Résultats des tests
- **htmlcov/index.html** : Rapport de couverture

---

## 🔍 Diagnostic

### Vérifier la Configuration Docker

```powershell
# Vérifier la configuration Docker
docker info | Select-String -Pattern "Proxy"

# Vérifier les réseaux Docker
docker network ls

# Vérifier les conteneurs Docker
docker ps
```

### Vérifier l'Infrastructure

```powershell
# Vérifier Kafka
docker ps | findstr "kafka"
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Vérifier PostgreSQL
docker ps | findstr "postgresql"
docker exec postgresql psql -U pmuser -d predictive_maintenance -c "SELECT 1;"
```

---

## ✅ Checklist

### Avant d'Exécuter les Tests

- [ ] Infrastructure Docker démarrée (Kafka, PostgreSQL)
- [ ] Python installé localement
- [ ] Dépendances Python installées
- [ ] Variables d'environnement configurées
- [ ] Réseau Docker configuré

### Après l'Exécution

- [ ] Tests exécutés avec succès
- [ ] Résultats validés
- [ ] Rapport de couverture généré (si demandé)
- [ ] Problèmes identifiés documentés

---

## 📚 Documentation

### Guides

- `SOLUTION_PROXY_DOCKER.md` : Solution rapide
- `FIX_DOCKER_PROXY.md` : Guide complet
- `TROUBLESHOOTING_DOCKER_NETWORK.md` : Dépannage réseau
- `FIX_DOCKER_PROXY_COMPLETE.md` : Ce guide

### Scripts

- `scripts/run-tests-local.ps1` : Tests locaux (recommandé)
- `scripts/run-tests-with-existing-infra.ps1` : Tests avec infrastructure existante
- `scripts/fix-docker-network.ps1` : Diagnostic réseau

---

## 🎯 Résumé

### Problème
- Proxy Docker mal configuré
- Docker Hub non accessible
- Timeout lors du téléchargement

### Solution Recommandée
- **Utiliser les tests locaux** avec l'infrastructure existante
- **Pas besoin de Docker Hub** pour les images Python
- **Utiliser l'infrastructure Docker existante** pour Kafka et PostgreSQL

### Commandes

```powershell
# 1. Démarrer l'infrastructure
cd infrastructure
docker-compose up -d

# 2. Exécuter les tests
cd ../services/preprocessing
.\scripts\run-tests-local.ps1
```

---

**Solution rapide** : Utilisez `.\scripts\run-tests-local.ps1` pour exécuter les tests sans problème de proxy !

