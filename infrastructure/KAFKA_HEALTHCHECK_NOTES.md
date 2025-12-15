# Notes sur le Health Check Kafka

## Problème Observé

Kafka peut afficher un statut "unhealthy" pendant les premières minutes après le démarrage, même si le service fonctionne correctement.

## Causes

1. **Démarrage lent** : Kafka prend du temps à initialiser complètement (1-3 minutes)
2. **Dépendances** : Kafka doit attendre Zookeeper et initialiser les métadonnées
3. **Topics** : Les topics existants doivent être chargés
4. **Health check trop strict** : Les commandes admin (`kafka-broker-api-versions`, `kafka-topics`) peuvent échouer même si Kafka fonctionne

## Solution Implémentée

Le health check utilise maintenant un test simple avec `nc` (netcat) qui vérifie juste que le port 9092 est ouvert :

```yaml
healthcheck:
  test: ["CMD-SHELL", "nc -z localhost 9092 || exit 1"]
  interval: 30s
  timeout: 5s
  retries: 10
  start_period: 180s  # 3 minutes pour le démarrage initial
```

## Utilisation des Commandes Kafka

### Depuis l'extérieur du conteneur (depuis votre machine)
Utilisez `localhost:9092` :
```powershell
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
```

### Depuis l'intérieur du conteneur (entre services)
Utilisez `kafka:29092` :
```bash
# Dans un autre conteneur
kafka-topics --list --bootstrap-server kafka:29092
```

### Depuis l'intérieur du conteneur Kafka lui-même
Utilisez `localhost:9092` :
```bash
# Dans le conteneur Kafka
kafka-topics --list --bootstrap-server localhost:9092
```

## Vérification que Kafka Fonctionne

### Méthode 1 : Vérifier les logs
```powershell
docker-compose logs kafka | Select-String "started"
```
Rechercher : `"Kafka Server started"`

### Méthode 2 : Vérifier le processus
```powershell
docker exec -it kafka ps aux | grep kafka
```

### Méthode 3 : Tester la connexion depuis l'extérieur
```powershell
# Depuis votre machine (pas dans le conteneur)
Test-NetConnection localhost -Port 9092
```

### Méthode 4 : Lister les topics (depuis l'extérieur)
```powershell
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
```
**Note** : Cette commande peut timeout si Kafka vient juste de démarrer. Attendre 1-2 minutes.

## Comportement Normal

✅ **Normal** :
- Kafka démarre en 1-3 minutes
- Health check peut être "unhealthy" pendant le démarrage
- Les commandes admin peuvent timeout pendant le démarrage
- Une fois démarré, tout fonctionne normalement

❌ **Problème** :
- Kafka reste unhealthy après 5 minutes
- Les logs montrent des erreurs répétées
- Aucun topic n'est accessible après plusieurs minutes

## En Cas de Problème Persistant

1. **Vérifier Zookeeper** :
   ```powershell
   docker-compose ps zookeeper
   docker-compose logs zookeeper --tail=20
   ```

2. **Redémarrer Kafka** :
   ```powershell
   docker-compose restart kafka
   ```

3. **Redémarrer tout** :
   ```powershell
   docker-compose down
   docker-compose up -d
   ```

4. **Vérifier les logs détaillés** :
   ```powershell
   docker-compose logs kafka --tail=100
   ```

## Notes Importantes

- Les **applications externes** utiliseront `localhost:9092` pour se connecter à Kafka
- Les **services dans Docker** utiliseront `kafka:29092` pour se connecter
- Le health check "unhealthy" temporaire est **normal** et ne bloque pas l'utilisation de Kafka
- Une fois Kafka complètement démarré, tous les topics sont accessibles et fonctionnels

