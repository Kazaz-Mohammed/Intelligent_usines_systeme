# Guide de Test - Déploiement Kubernetes

## Vue d'ensemble

Ce guide explique comment tester le déploiement Kubernetes de la plateforme de maintenance prédictive.

## Options de test

### Option 1 : Minikube (Recommandé pour débutants)

Minikube crée un cluster Kubernetes local dans une VM.

#### Installation
```bash
# Windows (avec Chocolatey)
choco install minikube

# Ou télécharger depuis: https://minikube.sigs.k8s.io/docs/start/
```

#### Configuration
```bash
cd infrastructure/kubernetes
chmod +x scripts/*.sh
./scripts/setup-minikube.sh
```

#### Avantages
- Simple à utiliser
- Bon pour développement local
- Supporte tous les features Kubernetes

#### Inconvénients
- Nécessite une VM (plus lourd)
- Plus lent que kind

### Option 2 : Kind (Kubernetes in Docker)

Kind crée un cluster Kubernetes dans des conteneurs Docker.

#### Installation
```bash
# Windows (avec Chocolatey)
choco install kind

# Ou télécharger depuis: https://kind.sigs.k8s.io/docs/user/quick-start/
```

#### Configuration
```bash
cd infrastructure/kubernetes
chmod +x scripts/*.sh
./scripts/setup-kind.sh
```

#### Avantages
- Plus léger que minikube
- Utilise Docker directement
- Rapide

#### Inconvénients
- Moins de features que minikube
- Nécessite Docker Desktop

## Étapes de test

### 1. Préparer les images Docker

Avant de déployer, vous devez builder les images Docker :

```bash
# Pour chaque service
cd services/ingestion-iiot
docker build -t predictive-maintenance/ingestion-iiot:latest .

cd ../preprocessing
docker build -t predictive-maintenance/preprocessing:latest .

cd ../extraction-features
docker build -t predictive-maintenance/extraction-features:latest .

cd ../detection-anomalies
docker build -t predictive-maintenance/detection-anomalies:latest .

cd ../prediction-rul
docker build -t predictive-maintenance/prediction-rul:latest .

cd ../orchestrateur-maintenance
docker build -t predictive-maintenance/orchestrateur-maintenance:latest .

cd ../dashboard-monitoring
docker build -t predictive-maintenance/dashboard-monitoring:latest .
```

**Pour minikube** : Les images doivent être dans le registry de minikube :
```bash
eval $(minikube docker-env)
# Puis builder les images (elles seront dans minikube)
```

**Pour kind** : Charger les images dans kind :
```bash
kind load docker-image predictive-maintenance/ingestion-iiot:latest --name predictive-maintenance
# Répéter pour chaque service
```

### 2. Créer les secrets

```bash
cd infrastructure/kubernetes
cp secrets/secrets-template.yaml secrets/secrets.yaml
# Éditer secrets/secrets.yaml avec vos valeurs
```

### 3. Déployer

```bash
cd infrastructure/kubernetes
./scripts/deploy-all.sh
```

### 4. Tester le déploiement

```bash
./scripts/test-deployment.sh
```

## Vérifications manuelles

### Vérifier les pods
```bash
kubectl get pods -n predictive-maintenance
```

### Vérifier les services
```bash
kubectl get services -n predictive-maintenance
```

### Vérifier les logs
```bash
# Logs d'un service spécifique
kubectl logs -f deployment/ingestion-iiot -n predictive-maintenance

# Logs de tous les pods
kubectl logs -f -l app=ingestion-iiot -n predictive-maintenance
```

### Tester un endpoint
```bash
# Port-forward
kubectl port-forward -n predictive-maintenance service/dashboard-monitoring-service 8086:8086

# Dans un autre terminal
curl http://localhost:8086/health
```

### Vérifier les ressources
```bash
# Utilisation des ressources
kubectl top pods -n predictive-maintenance

# Détails d'un pod
kubectl describe pod <pod-name> -n predictive-maintenance
```

## Tests d'intégration

### Test 1 : Vérifier la santé de tous les services
```bash
for service in ingestion-iiot preprocessing extraction-features detection-anomalies prediction-rul orchestrateur-maintenance dashboard-monitoring; do
    echo "Testing $service..."
    kubectl port-forward -n predictive-maintenance service/${service}-service 8080:8080 > /dev/null 2>&1 &
    sleep 2
    curl -f http://localhost:8080/health && echo "✅ $service OK" || echo "❌ $service FAILED"
    kill %1 2>/dev/null || true
done
```

### Test 2 : Vérifier la communication Kafka
```bash
# Vérifier que Kafka est accessible
kubectl exec -it -n predictive-maintenance deployment/kafka -- kafka-topics --list --bootstrap-server localhost:9092
```

### Test 3 : Vérifier la base de données
```bash
# Se connecter à PostgreSQL
kubectl exec -it -n predictive-maintenance deployment/postgresql -- psql -U pmuser -d predictive_maintenance -c "\dt"
```

## Dépannage

### Pods en CrashLoopBackOff
```bash
# Voir les logs
kubectl logs <pod-name> -n predictive-maintenance

# Voir les événements
kubectl describe pod <pod-name> -n predictive-maintenance
```

### Services non accessibles
```bash
# Vérifier les endpoints
kubectl get endpoints -n predictive-maintenance

# Vérifier les services
kubectl describe service <service-name> -n predictive-maintenance
```

### Problèmes de ressources
```bash
# Vérifier l'utilisation
kubectl top nodes
kubectl top pods -n predictive-maintenance

# Augmenter les ressources dans minikube
minikube config set memory 16384
minikube config set cpus 8
minikube stop
minikube start
```

## Nettoyage

### Supprimer le déploiement
```bash
cd infrastructure/kubernetes
./scripts/undeploy-all.sh
```

### Supprimer minikube
```bash
minikube delete
```

### Supprimer kind
```bash
kind delete cluster --name predictive-maintenance
```

## Notes importantes

1. **Ressources** : Assurez-vous d'avoir suffisamment de ressources (RAM, CPU)
2. **Images** : Les images doivent être disponibles dans le cluster
3. **Secrets** : Créer les secrets avant le déploiement
4. **Storage** : Vérifier que le StorageClass est disponible
5. **Ingress** : Installer un Ingress Controller si nécessaire

## Prochaines étapes

Une fois le déploiement testé et validé :
- Configurer le monitoring (Prometheus, Grafana)
- Mettre en place les backups
- Configurer le scaling automatique (HPA)
- Sécuriser avec RBAC et Network Policies

