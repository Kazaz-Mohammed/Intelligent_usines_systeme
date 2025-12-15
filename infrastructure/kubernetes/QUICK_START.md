# Quick Start - Déploiement Kubernetes

## ⚠️ Important pour Windows

Si vous êtes sur Windows et rencontrez une erreur avec VirtualBox/Hyper-V, consultez [WINDOWS_SETUP.md](WINDOWS_SETUP.md) pour les solutions.

**Recommandation Windows**: Utilisez Docker Desktop + Kind (plus simple)

## Démarrage rapide avec Minikube

### 1. Installer Minikube

**Windows (PowerShell en admin):**
```powershell
choco install minikube
```

**Ou télécharger:** https://minikube.sigs.k8s.io/docs/start/

### 2. Démarrer Minikube

**Sur Windows (avec Hyper-V):**
```bash
./scripts/setup-minikube-windows.sh
```

**Sur Linux/Mac:**
```bash
./scripts/setup-minikube.sh
```

**Ou manuellement:**
```bash
# Windows
minikube start --driver=hyperv --memory=8192 --cpus=4

# Linux/Mac
minikube start --driver=docker --memory=8192 --cpus=4
# ou
minikube start --driver=virtualbox --memory=8192 --cpus=4

minikube addons enable ingress
```

### 3. Configurer Docker pour Minikube

```bash
eval $(minikube docker-env)
```

### 4. Builder les images Docker

```bash
# Depuis la racine du projet
cd services/ingestion-iiot && docker build -t predictive-maintenance/ingestion-iiot:latest . && cd ../..
cd services/preprocessing && docker build -t predictive-maintenance/preprocessing:latest . && cd ../..
cd services/extraction-features && docker build -t predictive-maintenance/extraction-features:latest . && cd ../..
cd services/detection-anomalies && docker build -t predictive-maintenance/detection-anomalies:latest . && cd ../..
cd services/prediction-rul && docker build -t predictive-maintenance/prediction-rul:latest . && cd ../..
cd services/orchestrateur-maintenance && docker build -t predictive-maintenance/orchestrateur-maintenance:latest . && cd ../..
cd services/dashboard-monitoring && docker build -t predictive-maintenance/dashboard-monitoring:latest . && cd ../..
```

### 5. Créer les secrets

```bash
cd infrastructure/kubernetes
cp secrets/secrets-template.yaml secrets/secrets.yaml
# Éditer secrets/secrets.yaml avec vos valeurs (au minimum postgresql-password)
```

### 6. Déployer

```bash
chmod +x scripts/*.sh
./scripts/deploy-all.sh
```

### 7. Vérifier

```bash
# Voir les pods
kubectl get pods -n predictive-maintenance

# Health check
./scripts/health-check.sh

# Tester le dashboard
kubectl port-forward -n predictive-maintenance service/dashboard-monitoring-service 8086:8086
# Ouvrir http://localhost:8086 dans le navigateur
```

## Démarrage rapide avec Kind (RECOMMANDÉ pour Windows)

### 1. Prérequis

- **Docker Desktop** installé et démarré
- **Kind** installé:
  ```powershell
  choco install kind
  ```

### 2. Créer le cluster

```bash
cd infrastructure/kubernetes
chmod +x scripts/*.sh
./scripts/setup-kind.sh
```

### 3. Builder les images

```bash
# Builder toutes les images
./scripts/build-all-images.sh
```

### 4. Charger les images dans kind

```bash
# Charger toutes les images automatiquement
./scripts/load-images-kind.sh
```

**Ou manuellement:**
```bash
kind load docker-image predictive-maintenance/ingestion-iiot:latest --name predictive-maintenance
kind load docker-image predictive-maintenance/preprocessing:latest --name predictive-maintenance
kind load docker-image predictive-maintenance/extraction-features:latest --name predictive-maintenance
kind load docker-image predictive-maintenance/detection-anomalies:latest --name predictive-maintenance
kind load docker-image predictive-maintenance/prediction-rul:latest --name predictive-maintenance
kind load docker-image predictive-maintenance/orchestrateur-maintenance:latest --name predictive-maintenance
kind load docker-image predictive-maintenance/dashboard-monitoring:latest --name predictive-maintenance
```

### 4. Déployer

```bash
./scripts/deploy-all.sh
```

## Accès aux services

### Port-forward manuel

```bash
# Dashboard
kubectl port-forward -n predictive-maintenance service/dashboard-monitoring-service 8086:8086

# API Ingestion
kubectl port-forward -n predictive-maintenance service/ingestion-iiot-service 8081:8081

# API Détection Anomalies
kubectl port-forward -n predictive-maintenance service/detection-anomalies-service 8084:8084
```

### Via Ingress (si configuré)

Ajouter dans `/etc/hosts` (ou `C:\Windows\System32\drivers\etc\hosts`):
```
127.0.0.1 api.predictive-maintenance.local
127.0.0.1 dashboard.predictive-maintenance.local
```

Puis accéder:
- API: http://api.predictive-maintenance.local/api/v1/...
- Dashboard: http://dashboard.predictive-maintenance.local

## Commandes utiles

```bash
# Voir tous les pods
kubectl get pods -n predictive-maintenance

# Voir les logs
kubectl logs -f deployment/dashboard-monitoring -n predictive-maintenance

# Redémarrer un service
kubectl rollout restart deployment/dashboard-monitoring -n predictive-maintenance

# Scale un service
kubectl scale deployment/dashboard-monitoring --replicas=3 -n predictive-maintenance

# Supprimer tout
./scripts/undeploy-all.sh
```

## Dépannage

### Pods ne démarrent pas
```bash
# Voir les événements
kubectl get events -n predictive-maintenance --sort-by='.lastTimestamp'

# Voir les logs
kubectl logs <pod-name> -n predictive-maintenance
```

### Services non accessibles
```bash
# Vérifier les endpoints
kubectl get endpoints -n predictive-maintenance

# Tester depuis un pod
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- curl http://dashboard-monitoring-service:8086/health
```

