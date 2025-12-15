# Résolution de Problèmes Docker

## Problème : Erreur de Connexion à Docker Hub

### Symptôme

```
ERROR: failed to solve: python:3.11-slim: failed to resolve source metadata for docker.io/library/python:3.11-slim: failed to do request: Head "https://registry-1.docker.io/v2/library/python/manifests/3.11-slim": proxyconnect tcp: dial tcp: lookup http.docker.internal on 192.168.65.7:53: read udp 192.168.65.6:54732->192.168.65.7:53: i/o timeout
```

### Causes Possibles

1. **Problème de proxy Docker** : Configuration proxy incorrecte
2. **Problème de réseau DNS** : DNS ne peut pas résoudre les noms
3. **Problème de connexion Internet** : Pas de connexion à Docker Hub
4. **Problème de configuration Docker Desktop** : Paramètres réseau incorrects

---

## Solutions

### Solution 1 : Vérifier la Configuration Proxy Docker

1. **Ouvrir Docker Desktop**
2. **Aller dans Settings > Resources > Proxies**
3. **Vérifier les paramètres de proxy** :
   - Si vous utilisez un proxy, configurez-le correctement
   - Si vous n'utilisez pas de proxy, désactivez-le
4. **Redémarrer Docker Desktop**

### Solution 2 : Vérifier la Configuration DNS

1. **Ouvrir Docker Desktop**
2. **Aller dans Settings > Resources > Network**
3. **Vérifier les paramètres DNS** :
   - DNS par défaut : 8.8.8.8, 8.8.4.4 (Google DNS)
   - Ou utiliser les DNS de votre FAI
4. **Redémarrer Docker Desktop**

### Solution 3 : Télécharger l'Image Manuellement

```powershell
# Télécharger l'image Python manuellement
docker pull python:3.11-slim

# Vérifier que l'image est téléchargée
docker images | grep python
```

### Solution 4 : Utiliser une Image Alternative (Mirroir)

Modifier le `Dockerfile.test` pour utiliser une image alternative :

```dockerfile
# Option 1 : Utiliser une image déjà téléchargée
FROM python:3.11

# Option 2 : Utiliser un registre alternatif (si disponible)
# FROM registry.cn-hangzhou.aliyuncs.com/library/python:3.11-slim
```

### Solution 5 : Désactiver le Proxy Docker

1. **Ouvrir Docker Desktop**
2. **Aller dans Settings > Resources > Proxies**
3. **Désactiver le proxy** si activé
4. **Redémarrer Docker Desktop**

### Solution 6 : Vérifier la Connexion Internet

```powershell
# Tester la connexion à Docker Hub
ping registry-1.docker.io

# Tester la résolution DNS
nslookup registry-1.docker.io

# Tester avec curl
curl -I https://registry-1.docker.io/v2/
```

### Solution 7 : Utiliser un VPN ou Modifier les Paramètres Réseau

Si vous êtes derrière un pare-feu d'entreprise :

1. **Utiliser un VPN** pour contourner les restrictions
2. **Configurer Docker pour utiliser un proxy d'entreprise**
3. **Contacter l'administrateur réseau** pour autoriser l'accès à Docker Hub

---

## Solution Recommandée : Télécharger l'Image Manuellement

### Étape 1 : Télécharger l'Image

```powershell
# Télécharger l'image Python
docker pull python:3.11-slim

# Si cela échoue, essayer sans proxy
docker pull --platform linux/amd64 python:3.11-slim
```

### Étape 2 : Vérifier l'Image

```powershell
# Vérifier que l'image est téléchargée
docker images python:3.11-slim
```

### Étape 3 : Réexécuter le Script

```powershell
# Réexécuter le script de test
.\scripts\run-tests-docker.ps1
```

---

## Solution Alternative : Utiliser une Image Locale

Si vous avez déjà une image Python sur votre système :

### Étape 1 : Vérifier les Images Disponibles

```powershell
# Lister toutes les images Python
docker images | grep python
```

### Étape 2 : Modifier le Dockerfile.test

Si vous avez une image Python disponible, modifiez le `Dockerfile.test` :

```dockerfile
# Utiliser une image Python déjà disponible
FROM python:3.11

# Ou utiliser une image spécifique
# FROM python:3.11.6-slim
```

### Étape 3 : Réexécuter le Script

```powershell
# Réexécuter le script de test
.\scripts\run-tests-docker.ps1
```

---

## Solution de Contournement : Utiliser l'Infrastructure Existante

Au lieu de créer de nouveaux conteneurs, utiliser l'infrastructure existante :

### Étape 1 : Vérifier l'Infrastructure Existante

```powershell
# Vérifier que l'infrastructure est démarrée
cd ..\..\infrastructure
docker-compose ps
```

### Étape 2 : Modifier docker-compose.test.yml

Utiliser les services existants au lieu de créer de nouveaux :

```yaml
services:
  preprocessing-test:
    # ...
    environment:
      - KAFKA_BOOTSTRAP_SERVERS=localhost:9092
      - DATABASE_HOST=localhost
      # ...
    network_mode: host  # Utiliser le réseau de l'hôte
```

### Étape 3 : Réexécuter les Tests

```powershell
# Exécuter les tests avec l'infrastructure existante
docker-compose -f docker-compose.test.yml run --rm preprocessing-test pytest tests/ -v
```

---

## Vérification Rapide

### Test 1 : Connexion Docker Hub

```powershell
# Tester la connexion
docker pull hello-world
```

Si cela fonctionne, le problème est spécifique à l'image Python.

### Test 2 : Configuration Docker

```powershell
# Vérifier la configuration Docker
docker info
```

Vérifier les paramètres de proxy et DNS.

### Test 3 : Résolution DNS

```powershell
# Tester la résolution DNS
nslookup registry-1.docker.io
```

Si cela échoue, le problème est lié au DNS.

---

## Prochaines Étapes

1. **Essayer Solution 1** : Vérifier la configuration proxy Docker
2. **Essayer Solution 2** : Vérifier la configuration DNS
3. **Essayer Solution 3** : Télécharger l'image manuellement
4. **Essayer Solution Alternative** : Utiliser une image locale
5. **Essayer Solution de Contournement** : Utiliser l'infrastructure existante

---

## Support

Si le problème persiste :

1. **Vérifier les logs Docker** : `docker logs <container-id>`
2. **Vérifier la configuration Docker Desktop** : Settings > Resources
3. **Contacter le support** : Fournir les logs d'erreur complets

---

**Note** : Si vous êtes dans un environnement d'entreprise avec un pare-feu, il peut être nécessaire de configurer un proxy ou d'obtenir l'autorisation de l'administrateur réseau.

