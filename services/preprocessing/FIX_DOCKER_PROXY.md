# Solution : Problème de Proxy Docker

## 🔴 Problème Identifié

Docker Desktop essaie d'utiliser un proxy (`http.docker.internal:3128`) mais ne peut pas le résoudre :
```
proxyconnect tcp: dial tcp: lookup http.docker.internal on 192.168.65.7:53: read udp 192.168.65.6:46634->192.168.65.7:53: i/o timeout
```

## ✅ Solutions

### Solution 1 : Désactiver le Proxy dans Docker Desktop (Recommandé)

1. **Ouvrir Docker Desktop**
2. **Aller dans Settings (⚙️)**
3. **Aller dans Resources > Proxies**
4. **Désactiver le proxy** :
   - Décocher "Manual proxy configuration"
   - Ou laisser vide les champs de proxy
5. **Appliquer les changements**
6. **Redémarrer Docker Desktop**
7. **Réessayer** :
   ```powershell
   docker pull python:3.11
   ```

### Solution 2 : Corriger la Configuration du Proxy

Si vous utilisez un proxy, configurez-le correctement :

1. **Ouvrir Docker Desktop**
2. **Aller dans Settings > Resources > Proxies**
3. **Configurer le proxy manuellement** :
   - **Web Server (HTTP Proxy)** : `http://votre-proxy:port`
   - **Secure Web Server (HTTPS Proxy)** : `http://votre-proxy:port`
   - **No Proxy** : `localhost,127.0.0.1,*.local`
4. **Appliquer les changements**
5. **Redémarrer Docker Desktop**

### Solution 3 : Utiliser une Image Déjà Téléchargée

Si vous avez déjà des images Python localement :

1. **Vérifier les images disponibles** :
   ```powershell
   docker images | grep python
   ```

2. **Utiliser une image existante** :
   - Si vous avez `python:3.11`, utilisez-la
   - Si vous avez `python:3.10`, utilisez-la temporairement

3. **Modifier le Dockerfile** :
   ```dockerfile
   FROM python:3.11
   # ou
   FROM python:3.10
   ```

### Solution 4 : Télécharger l'Image Manuellement (Sans Proxy)

1. **Désactiver le proxy temporairement** :
   ```powershell
   # Dans PowerShell
   $env:HTTP_PROXY=""
   $env:HTTPS_PROXY=""
   ```

2. **Télécharger l'image** :
   ```powershell
   docker pull python:3.11
   ```

3. **Vérifier que l'image existe** :
   ```powershell
   docker images | grep python
   ```

4. **Construire l'image de test** :
   ```powershell
   cd services/preprocessing
   docker build -f Dockerfile.test -t preprocessing-test:latest .
   ```

### Solution 5 : Utiliser une Image Alternative (Alpine)

Alpine est plus petite et souvent plus rapide à télécharger :

1. **Modifier le Dockerfile** :
   ```dockerfile
   FROM python:3.11-alpine
   ```

2. **Mettre à jour les commandes d'installation** :
   ```dockerfile
   RUN apk add --no-cache \
       gcc \
       g++ \
       postgresql-dev \
       musl-dev \
       netcat-openbsd \
       linux-headers
   ```

3. **Construire l'image** :
   ```powershell
   docker build -f Dockerfile.test -t preprocessing-test:latest .
   ```

### Solution 6 : Utiliser une Image Locale (Docker Hub Offline)

Si Docker Hub n'est pas accessible, utilisez une image locale :

1. **Vérifier les images disponibles** :
   ```powershell
   docker images
   ```

2. **Télécharger l'image depuis un autre ordinateur** :
   - Si vous avez accès à un autre ordinateur avec Docker
   - Télécharger l'image : `docker pull python:3.11`
   - Sauvegarder l'image : `docker save python:3.11 > python-3.11.tar`
   - Charger l'image : `docker load < python-3.11.tar`

3. **Utiliser l'image locale** :
   ```dockerfile
   FROM python:3.11
   ```

---

## 🔧 Configuration Rapide

### Désactiver le Proxy (Méthode Rapide)

1. **Ouvrir Docker Desktop**
2. **Settings > Resources > Proxies**
3. **Décocher "Manual proxy configuration"**
4. **Appliquer et Redémarrer**

### Vérifier la Configuration

```powershell
# Vérifier la configuration Docker
docker info | Select-String -Pattern "Proxy"

# Vérifier les images disponibles
docker images | grep python

# Tester le téléchargement
docker pull python:3.11
```

---

## 📋 Checklist

### Avant de Réessayer

- [ ] Proxy désactivé dans Docker Desktop
- [ ] Docker Desktop redémarré
- [ ] Connectivité Internet vérifiée
- [ ] DNS configuré correctement
- [ ] Image Python téléchargée (optionnel)

### Après la Configuration

- [ ] Image Python téléchargée avec succès
- [ ] Image de test construite avec succès
- [ ] Tests exécutés avec succès
- [ ] Résultats validés

---

## 🚀 Solution Recommandée (Étape par Étape)

### Étape 1 : Désactiver le Proxy

1. **Ouvrir Docker Desktop**
2. **Settings > Resources > Proxies**
3. **Décocher "Manual proxy configuration"**
4. **Appliquer les changements**
5. **Redémarrer Docker Desktop**

### Étape 2 : Vérifier la Connectivité

```powershell
# Tester le téléchargement
docker pull python:3.11

# Si ça fonctionne, continuer
# Si ça ne fonctionne pas, essayer une autre solution
```

### Étape 3 : Construire l'Image de Test

```powershell
# Se placer dans le répertoire
cd services/preprocessing

# Construire l'image
docker build -f Dockerfile.test -t preprocessing-test:latest .
```

### Étape 4 : Exécuter les Tests

```powershell
# Exécuter les tests
.\scripts\run-tests-docker.ps1
```

---

## 🎯 Solution Alternative : Utiliser l'Infrastructure Existante

Si Docker Hub n'est pas accessible, utilisez l'infrastructure existante :

1. **Utiliser l'infrastructure Docker existante** :
   ```powershell
   cd infrastructure
   docker-compose up -d
   ```

2. **Exécuter les tests directement** (sans Docker) :
   ```powershell
   cd services/preprocessing
   pip install -r requirements.txt
   pytest tests/ -v
   ```

3. **Ou utiliser l'infrastructure existante pour les tests d'intégration** :
   ```powershell
   # Démarrer l'infrastructure
   cd infrastructure
   docker-compose up -d

   # Exécuter les tests d'intégration
   cd ../services/preprocessing
   pytest tests/ -v -m integration
   ```

---

## ✅ Résumé

### Problème
- Proxy Docker mal configuré
- `http.docker.internal` ne peut pas être résolu
- Timeout lors du téléchargement d'images

### Solutions
1. ✅ **Désactiver le proxy** (recommandé)
2. ✅ **Corriger la configuration du proxy**
3. ✅ **Utiliser une image déjà téléchargée**
4. ✅ **Utiliser une image Alpine**
5. ✅ **Utiliser l'infrastructure existante**

### Recommandation
- **Désactiver le proxy dans Docker Desktop** si vous n'en avez pas besoin
- **Utiliser python:3.11** au lieu de python:3.11-slim
- **Utiliser l'infrastructure existante** pour les tests d'intégration

---

**Problème résolu ?** Après avoir désactivé le proxy, réessayez `docker pull python:3.11` !

