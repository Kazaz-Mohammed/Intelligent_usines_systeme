# Déploiement Kubernetes

## Description

Configurations Kubernetes pour déployer la plateforme de maintenance prédictive sur un cluster Kubernetes.

## Structure

```
kubernetes/
├── namespace.yaml
├── configmaps/
│   ├── kafka-config.yaml
│   ├── postgresql-config.yaml
│   └── services-config.yaml
├── secrets/
│   └── secrets-template.yaml
├── postgresql/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── pvc.yaml
├── kafka/
│   ├── zookeeper-deployment.yaml
│   ├── zookeeper-service.yaml
│   ├── kafka-deployment.yaml
│   ├── kafka-service.yaml
│   └── topics-configmap.yaml
├── services/
│   ├── ingestion-iiot/
│   ├── preprocessing/
│   ├── extraction-features/
│   ├── detection-anomalies/
│   ├── prediction-rul/
│   ├── orchestrateur-maintenance/
│   └── dashboard-monitoring/
├── ingress/
│   └── ingress.yaml
└── README.md
```

## Prérequis

- Cluster Kubernetes (minikube, kind, ou cloud)
- kubectl configuré
- Helm (optionnel, pour certaines dépendances)

## Déploiement

### 1. Créer le namespace
```bash
kubectl apply -f namespace.yaml
```

### 2. Créer les secrets
```bash
# Copier et modifier secrets-template.yaml
cp secrets/secrets-template.yaml secrets/secrets.yaml
# Éditer avec vos valeurs
kubectl apply -f secrets/secrets.yaml
```

### 3. Déployer l'infrastructure
```bash
# PostgreSQL
kubectl apply -f postgresql/

# Kafka
kubectl apply -f kafka/
```

### 4. Déployer les services
```bash
# Tous les services
kubectl apply -f services/
```

### 5. Configurer l'ingress
```bash
kubectl apply -f ingress/
```

## État

🚧 **Phase 11 en cours - Configurations Kubernetes à créer**

