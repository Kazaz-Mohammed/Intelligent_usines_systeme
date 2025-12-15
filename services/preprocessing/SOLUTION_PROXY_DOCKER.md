# Solution Rapide : Problème de Proxy Docker

## 🔴 Problème

Docker ne peut pas télécharger l'image `python:3.11-slim` à cause d'un problème de proxy :
```
proxyconnect tcp: dial tcp: lookup http.docker.internal on 192.168.65.7:53: i/o timeout
```

## ✅ Solution Rapide (Recommandée)

### Option 1 : Désactiver le Proxy dans Docker Desktop

1. **Ouvrir Docker Desktop**
2. **Settings (⚙️) > Resources > Proxies**
3. **Décocher "Manual proxy configuration"**
4. **Appliquer les changements**
5. **Redémarrer Docker Desktop**
6. **Réessayer** :
   ```powershell
   docker pull python:3.11
   docker build -f Dockerfile.test -t preprocessing-test:latest .
   ```

### Option 2 : Utiliser les Tests Locaux (Sans Docker)

Si Docker Hub n'est pas accessible, utilisez les tests locaux :

```powershell
# 1. Démarrer l'infrastructure (si nécessaire)
cd infrastructure
docker-compose up -d

# 2. Exécuter les tests localement
cd ../services/preprocessing
.\scripts\run-tests-local.ps1
```

### Option 3 : Utiliser une Image Locale

Si vous avez déjà une image Python :

```powershell
# 1. Vérifier les images disponibles
docker images | grep python

# 2. Si vous avez python:3.11, utiliser le Dockerfile.test.local
docker build -f Dockerfile.test.local -t preprocessing-test:latest .
```

---

## 🚀 Solution Recommandée : Tests Locaux

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

```powershell
# Résultats dans le terminal
# Rapport de couverture : htmlcov/index.html
start htmlcov/index.html
```

---

## 📋 Avantages des Tests Locaux

1. ✅ **Pas de problème de proxy** : Utilise directement Python
2. ✅ **Plus rapide** : Pas besoin de construire une image Docker
3. ✅ **Plus simple** : Utilise l'infrastructure existante
4. ✅ **Même résultats** : Tests identiques

---

## 🔧 Correction du Problème de Proxy

### Méthode 1 : Désactiver le Proxy

1. **Ouvrir Docker Desktop**
2. **Settings > Resources > Proxies**
3. **Décocher "Manual proxy configuration"**
4. **Appliquer et Redémarrer**

### Méthode 2 : Corriger le Proxy

1. **Ouvrir Docker Desktop**
2. **Settings > Resources > Proxies**
3. **Configurer le proxy correctement** :
   - **Web Server (HTTP Proxy)** : `http://votre-proxy:port`
   - **Secure Web Server (HTTPS Proxy)** : `http://votre-proxy:port`
   - **No Proxy** : `localhost,127.0.0.1,*.local`
4. **Appliquer et Redémarrer**

---

## 📚 Documentation Complète

Pour plus de détails, consultez :
- `FIX_DOCKER_PROXY.md` : Guide complet
- `TROUBLESHOOTING_DOCKER_NETWORK.md` : Dépannage réseau
- `DOCKER_TESTING_GUIDE.md` : Guide Docker

---

## ✅ Résumé

### Problème
- Proxy Docker mal configuré
- Docker Hub non accessible
- Timeout lors du téléchargement

### Solutions
1. ✅ **Désactiver le proxy** (recommandé)
2. ✅ **Utiliser les tests locaux** (rapide)
3. ✅ **Utiliser une image locale** (alternative)

### Recommandation
- **Utiliser les tests locaux** si Docker Hub n'est pas accessible
- **Désactiver le proxy** si vous n'en avez pas besoin
- **Utiliser l'infrastructure existante** pour les tests d'intégration

---

**Solution rapide** : Utilisez `.\scripts\run-tests-local.ps1` pour exécuter les tests sans Docker !

