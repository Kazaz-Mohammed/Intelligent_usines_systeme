# Dépannage - Infrastructure Docker

## Problèmes Courants et Solutions

### 1. Timeout lors du téléchargement des images Docker

**Symptôme** :
```
failed to copy: httpReadSeeker: failed open: failed to do request: 
Get "https://production.cloudflare.docker.com/...": net/http: TLS handshake timeout
```

**Solutions** :

#### Solution A : Réessayer
```powershell
# Arrêter les conteneurs en cours
docker-compose down

# Réessayer le pull
docker-compose pull

# Puis démarrer
docker-compose up -d
```

#### Solution B : Vérifier la connexion réseau
```powershell
# Tester la connexion à Docker Hub
ping registry-1.docker.io

# Vérifier les paramètres proxy dans Docker Desktop
# Settings > Resources > Proxies
```

#### Solution C : Utiliser un mirror ou configurer Docker
```powershell
# Vérifier si Docker Desktop est en cours d'exécution
docker info

# Redémarrer Docker Desktop si nécessaire
```

#### Solution D : Télécharger les images une par une
```powershell
docker pull timescale/timescaledb:latest-pg16
docker pull confluentinc/cp-zookeeper:7.5.0
docker pull confluentinc/cp-kafka:7.5.0
docker pull influxdb:2.7
docker pull redis:7-alpine
docker pull minio/minio:latest
```

#### Solution E : Augmenter le timeout Docker
Dans Docker Desktop :
- Settings > Docker Engine
- Ajouter :
```json
{
  "max-concurrent-downloads": 3,
  "max-concurrent-uploads": 5
}
```
- Apply & Restart

### 2. Erreur "port already in use"

**Symptôme** : Erreur lors du démarrage indiquant qu'un port est déjà utilisé

**Solution** :
```powershell
# Vérifier quel processus utilise le port
netstat -ano | findstr :5432

# Option A : Arrêter le processus
# Option B : Modifier les ports dans docker-compose.yml
```

### 3. Avertissement "version is obsolete"

**Solution** : Déjà corrigé - la ligne `version: '3.8'` a été retirée

### 4. Erreur de permission sur les volumes (Linux/Mac)

**Solution** :
```bash
sudo chown -R $USER:$USER volumes/
```

### 5. Service ne démarre pas

**Vérifier les logs** :
```powershell
docker-compose logs [service-name]
docker-compose logs postgresql
docker-compose logs kafka
```

### 6. Health check échoue

**Solution** : Attendre plus longtemps (les services peuvent prendre du temps à démarrer)
```powershell
# Vérifier l'état après 30 secondes
Start-Sleep -Seconds 30
docker-compose ps
```

### 7. Problème de mémoire insuffisante

**Solution** :
- Docker Desktop : Settings > Resources > Memory
- Augmenter la RAM allouée (minimum 4GB recommandé)

### 8. Images corrompues

**Solution** :
```powershell
# Supprimer les images corrompues
docker-compose down --rmi all

# Redémarrer le téléchargement
docker-compose pull
docker-compose up -d
```

## Commandes de Diagnostic

### Vérifier l'état de Docker
```powershell
docker info
docker ps -a
docker images
```

### Vérifier les réseaux
```powershell
docker network ls
docker network inspect predictive-maintenance-network
```

### Vérifier les volumes
```powershell
docker volume ls
docker volume inspect [volume-name]
```

### Nettoyer Docker (si nécessaire)
```powershell
# Supprimer les conteneurs arrêtés
docker container prune

# Supprimer les images non utilisées
docker image prune -a

# Nettoyage complet (⚠️ supprime tout)
docker system prune -a --volumes
```

## Étapes de Récupération Complète

Si rien ne fonctionne :

1. **Arrêter tout** :
```powershell
docker-compose down -v
```

2. **Nettoyer Docker** :
```powershell
docker system prune -a
```

3. **Redémarrer Docker Desktop**

4. **Relancer le pull** :
```powershell
cd infrastructure
docker-compose pull
```

5. **Démarrer les services** :
```powershell
docker-compose up -d
```

6. **Vérifier l'état** :
```powershell
docker-compose ps
```

## Support

Si le problème persiste :
1. Vérifier les logs détaillés : `docker-compose logs`
2. Vérifier la connexion Internet
3. Vérifier que Docker Desktop fonctionne correctement
4. Consulter la documentation Docker : https://docs.docker.com

