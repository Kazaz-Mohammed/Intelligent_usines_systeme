# Correction Rapide : Problème de Connexion Docker Hub

## Problème

```
ERROR: failed to solve: python:3.11-slim: failed to resolve source metadata for docker.io/library/python:3.11-slim
```

## Solution Rapide (3 Étapes)

### Étape 1 : Télécharger l'Image Manuellement

```powershell
# Télécharger l'image Python
docker pull python:3.11-slim

# Si cela échoue, essayer :
docker pull --platform linux/amd64 python:3.11-slim
```

### Étape 2 : Vérifier l'Image

```powershell
# Vérifier que l'image est téléchargée
docker images python:3.11-slim
```

### Étape 3 : Réexécuter les Tests

```powershell
# Réexécuter le script de test
.\scripts\run-tests-docker.ps1
```

---

## Si cela ne fonctionne pas

### Option 1 : Utiliser une Image Alternative

Modifier `Dockerfile.test` :

```dockerfile
# Utiliser python:3.11 au lieu de python:3.11-slim
FROM python:3.11
```

### Option 2 : Vérifier la Configuration Proxy

1. Ouvrir Docker Desktop
2. Settings > Resources > Proxies
3. Désactiver le proxy si activé
4. Redémarrer Docker Desktop

### Option 3 : Vérifier la Configuration DNS

1. Ouvrir Docker Desktop
2. Settings > Resources > Network
3. Configurer DNS : 8.8.8.8, 8.8.4.4
4. Redémarrer Docker Desktop

---

## Script de Diagnostic

```powershell
# Exécuter le script de diagnostic
.\scripts\test-docker-connection.ps1
```

Ce script va :
- ✅ Vérifier Docker
- ✅ Vérifier Docker Desktop
- ✅ Tester la connexion à Docker Hub
- ✅ Télécharger l'image Python
- ✅ Vérifier l'image téléchargée

---

## Documentation Complète

Pour plus de détails, consultez :
- `TROUBLESHOOTING_DOCKER.md` : Guide complet de résolution de problèmes
- `QUICK_FIX_DOCKER.md` : Ce guide rapide

---

**Solution la plus rapide** : Télécharger l'image manuellement avec `docker pull python:3.11-slim`

