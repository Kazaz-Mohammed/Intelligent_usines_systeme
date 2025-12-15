# Résolution des Problèmes de Réseau Docker

## Date : 13 novembre 2025

---

## 🔴 Problème : Erreur de Connectivité Docker

### Erreur

```
ERROR: failed to solve: python:3.11-slim: failed to resolve source metadata for docker.io/library/python:3.11-slim: failed to do request: Head "https://registry-1.docker.io/v2/library/python/manifests/3.11-slim": proxyconnect tcp: dial tcp: lookup http.docker.internal on 192.168.65.7:53: read udp 192.168.65.6:54732->192.168.65.7:53: i/o timeout
```

### Cause

Problème de connectivité réseau/DNS avec Docker Desktop :
1. **Problème DNS** : Docker ne peut pas résoudre `http.docker.internal`
2. **Problème de proxy** : Configuration proxy incorrecte
3. **Problème de réseau** : Réseau Docker non configuré correctement
4. **Problème de connectivité** : Pas d'accès à Docker Hub

---

## 🔧 Solutions

### Solution 1 : Vérifier la Configuration Docker Desktop

1. **Ouvrir Docker Desktop**
2. **Aller dans Settings > Resources > Network**
3. **Vérifier les paramètres réseau** :
   - DNS : Configuré correctement
   - Proxy : Désactivé ou configuré correctement
   - Network : Bridge configuré

4. **Appliquer les changements**
5. **Redémarrer Docker Desktop**

### Solution 2 : Configurer le Proxy Docker

Si vous utilisez un proxy, configurez-le dans Docker Desktop :

1. **Ouvrir Docker Desktop**
2. **Aller dans Settings > Resources > Proxies**
3. **Configurer le proxy** :
   - Web Server (HTTP Proxy) : `http://proxy.example.com:8080`
   - Secure Web Server (HTTPS Proxy) : `https://proxy.example.com:8080`
   - No Proxy : `localhost,127.0.0.1`

4. **Appliquer les changements**
5. **Redémarrer Docker Desktop**

### Solution 3 : Vérifier la Connectivité Internet

1. **Vérifier la connectivité Internet** :
   ```powershell
   ping registry-1.docker.io
   ```

2. **Vérifier la résolution DNS** :
   ```powershell
   nslookup registry-1.docker.io
   ```

3. **Vérifier l'accès à Docker Hub** :
   ```powershell
   curl https://registry-1.docker.io/v2/
   ```

### Solution 4 : Utiliser un Miroir Docker Hub

Si Docker Hub n'est pas accessible, utilisez un miroir :

1. **Configurer le miroir dans Docker Desktop** :
   - Settings > Docker Engine
   - Ajouter :
     ```json
     {
       "registry-mirrors": [
         "https://mirror.gcr.io"
       ]
     }
     ```

2. **Appliquer les changements**
3. **Redémarrer Docker Desktop**

### Solution 5 : Utiliser une Image Locale

Si les images Docker ne sont pas accessibles, utilisez une image locale :

1. **Télécharger l'image manuellement** :
   ```powershell
   docker pull python:3.11-slim
   ```

2. **Vérifier que l'image existe** :
   ```powershell
   docker images | grep python
   ```

3. **Construire l'image de test** :
   ```powershell
   docker build -f Dockerfile.test -t preprocessing-test:latest .
   ```

### Solution 6 : Réinitialiser Docker Desktop

Si rien ne fonctionne, réinitialiser Docker Desktop :

1. **Ouvrir Docker Desktop**
2. **Aller dans Troubleshoot > Reset to factory defaults**
3. **Confirmer la réinitialisation**
4. **Redémarrer Docker Desktop**

---

## 🔍 Diagnostic

### Vérifier la Configuration Docker

```powershell
# Vérifier la configuration Docker
docker info

# Vérifier les réseaux Docker
docker network ls

# Vérifier les images Docker
docker images

# Vérifier les conteneurs Docker
docker ps -a
```

### Vérifier la Connectivité

```powershell
# Vérifier la connectivité Internet
ping 8.8.8.8

# Vérifier la résolution DNS
nslookup docker.io

# Vérifier l'accès à Docker Hub
curl https://registry-1.docker.io/v2/
```

### Vérifier les Logs Docker

```powershell
# Vérifier les logs Docker Desktop
# Ouvrir Docker Desktop > Troubleshoot > View logs
```

---

## 🎯 Solutions Rapides

### Solution Rapide 1 : Redémarrer Docker Desktop

```powershell
# Arrêter Docker Desktop
# Démarrer Docker Desktop
# Attendre que Docker soit prêt
# Réessayer
```

### Solution Rapide 2 : Vérifier les Paramètres Réseau

```powershell
# Vérifier les paramètres réseau
ipconfig /all

# Vérifier les paramètres DNS
nslookup docker.io
```

### Solution Rapide 3 : Utiliser une Image Alternative

Si `python:3.11-slim` n'est pas accessible, utilisez une image alternative :

```dockerfile
# Utiliser une image alternative
FROM python:3.11-alpine

# Ou utiliser une image locale
FROM python:3.11
```

---

## 📋 Checklist de Diagnostic

### Vérifications de Base

- [ ] Docker Desktop démarré
- [ ] Connectivité Internet fonctionnelle
- [ ] DNS configuré correctement
- [ ] Proxy configuré correctement (si utilisé)
- [ ] Réseau Docker configuré correctement

### Vérifications Avancées

- [ ] Accès à Docker Hub fonctionnel
- [ ] Images Docker téléchargées
- [ ] Réseaux Docker créés
- [ ] Conteneurs Docker fonctionnels
- [ ] Logs Docker sans erreur

---

## 🚀 Solution Recommandée

### Étape 1 : Vérifier Docker Desktop

1. **Ouvrir Docker Desktop**
2. **Vérifier que Docker est prêt** (icône Docker dans la barre des tâches)
3. **Vérifier les paramètres réseau** (Settings > Resources > Network)

### Étape 2 : Tester la Connectivité

```powershell
# Tester la connectivité Docker Hub
docker pull hello-world

# Si ça fonctionne, tester Python
docker pull python:3.11-slim
```

### Étape 3 : Configurer le Proxy (si nécessaire)

1. **Ouvrir Docker Desktop**
2. **Aller dans Settings > Resources > Proxies**
3. **Configurer le proxy** (si utilisé)
4. **Appliquer les changements**
5. **Redémarrer Docker Desktop**

### Étape 4 : Réessayer

```powershell
# Réessayer la construction de l'image
cd services/preprocessing
docker build -f Dockerfile.test -t preprocessing-test:latest .
```

---

## 🔧 Configuration Alternative

### Utiliser une Image Alpine (plus petite, plus rapide)

```dockerfile
# Dockerfile pour les tests (version Alpine)
FROM python:3.11-alpine

# Installation des dépendances système
RUN apk add --no-cache \
    gcc \
    g++ \
    libpq-dev \
    netcat-openbsd

# Reste du Dockerfile...
```

### Utiliser une Image Locale

```dockerfile
# Dockerfile pour les tests (version locale)
FROM python:3.11

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Reste du Dockerfile...
```

---

## 📚 Ressources

### Documentation

- [Docker Desktop Network Settings](https://docs.docker.com/desktop/settings/windows/#network)
- [Docker Proxy Configuration](https://docs.docker.com/config/daemon/systemd/#httphttps-proxy)
- [Docker Hub Mirror](https://docs.docker.com/registry/recipes/mirror/)

### Commandes Utiles

```powershell
# Vérifier la configuration Docker
docker info

# Vérifier les réseaux Docker
docker network ls

# Vérifier les images Docker
docker images

# Vérifier les conteneurs Docker
docker ps -a

# Vérifier les logs Docker
docker logs <container-id>
```

---

## ✅ Résumé

### Problème
- Erreur de connectivité Docker Hub
- Problème DNS/proxy
- Timeout réseau

### Solutions
1. ✅ Vérifier Docker Desktop
2. ✅ Configurer le proxy (si nécessaire)
3. ✅ Vérifier la connectivité Internet
4. ✅ Utiliser un miroir Docker Hub
5. ✅ Utiliser une image locale
6. ✅ Réinitialiser Docker Desktop

### Recommandation
- Vérifier d'abord Docker Desktop et la connectivité Internet
- Configurer le proxy si nécessaire
- Utiliser une image Alpine si les problèmes persistent

---

**Problème résolu ?** Réessayez `.\scripts\run-tests-docker.ps1` après avoir appliqué les solutions !

