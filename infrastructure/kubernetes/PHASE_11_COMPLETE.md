# Phase 11 - Déploiement Kubernetes - COMPLÉTÉE ✅

## Résumé

La Phase 11 est **complétée** avec succès ! Toutes les configurations Kubernetes sont créées pour déployer la plateforme sur un cluster Kubernetes.

## 📊 Statistiques

### Fichiers créés
- **Namespace** : 1 fichier
- **ConfigMaps** : 3 fichiers
- **Secrets** : 1 template
- **PostgreSQL** : 3 fichiers (deployment, service, PVC)
- **Kafka** : 4 fichiers (zookeeper + kafka)
- **Services** : 14 fichiers (7 deployments + 7 services)
- **Ingress** : 1 fichier
- **Scripts** : 2 scripts

### Total : 29 fichiers Kubernetes

## ✅ Composants implémentés

### 1. Namespace ✅
- ✅ `namespace.yaml` - Namespace `predictive-maintenance`

### 2. ConfigMaps ✅
- ✅ `kafka-config.yaml` - Configuration Kafka
- ✅ `postgresql-config.yaml` - Configuration PostgreSQL
- ✅ `services-config.yaml` - URLs des services

### 3. Secrets ✅
- ✅ `secrets-template.yaml` - Template pour secrets

### 4. Infrastructure ✅
- ✅ **PostgreSQL** :
  - Deployment avec TimescaleDB
  - Service ClusterIP
  - PVC (20Gi)
- ✅ **Kafka** :
  - Zookeeper (Deployment + Service)
  - Kafka (Deployment + Service)

### 5. Services applicatifs ✅
- ✅ **IngestionIIoT** (Port 8081)
- ✅ **Preprocessing** (Port 8082)
- ✅ **ExtractionFeatures** (Port 8083)
- ✅ **DetectionAnomalies** (Port 8084)
- ✅ **PredictionRUL** (Port 8085)
- ✅ **OrchestrateurMaintenance** (Port 8087)
- ✅ **DashboardMonitoring** (Port 8086)

Chaque service a :
- Deployment avec replicas=2
- Service ClusterIP
- Health checks (liveness + readiness)
- Resource limits
- Variables d'environnement depuis ConfigMaps/Secrets

### 6. Ingress ✅
- ✅ `ingress.yaml` - Ingress avec 2 hosts :
  - `api.predictive-maintenance.local` - API REST
  - `dashboard.predictive-maintenance.local` - Dashboard

### 7. Scripts ✅
- ✅ `deploy-all.sh` - Déploiement complet
- ✅ `undeploy-all.sh` - Suppression complète

## 🚀 Déploiement

### Prérequis
```bash
# Cluster Kubernetes (minikube, kind, ou cloud)
kubectl cluster-info

# Vérifier l'accès
kubectl get nodes
```

### Déploiement complet
```bash
cd infrastructure/kubernetes

# 1. Créer les secrets (depuis template)
cp secrets/secrets-template.yaml secrets/secrets.yaml
# Éditer secrets/secrets.yaml avec vos valeurs

# 2. Déployer tout
chmod +x scripts/*.sh
./scripts/deploy-all.sh
```

### Vérification
```bash
# Voir les pods
kubectl get pods -n predictive-maintenance

# Voir les services
kubectl get services -n predictive-maintenance

# Voir les logs d'un service
kubectl logs -f deployment/ingestion-iiot -n predictive-maintenance
```

### Accès aux services
```bash
# Port-forward pour accès local
kubectl port-forward -n predictive-maintenance service/dashboard-monitoring-service 8086:8086

# Accès via Ingress (si configuré)
# http://api.predictive-maintenance.local/api/v1/...
# http://dashboard.predictive-maintenance.local
```

## 📋 Configuration des ressources

### Ressources par service

| Service | Replicas | Memory Request | CPU Request | Memory Limit | CPU Limit |
|---------|----------|----------------|-------------|--------------|-----------|
| IngestionIIoT | 2 | 256Mi | 100m | 512Mi | 500m |
| Preprocessing | 2 | 512Mi | 250m | 1Gi | 1000m |
| ExtractionFeatures | 2 | 512Mi | 250m | 1Gi | 1000m |
| DetectionAnomalies | 2 | 1Gi | 500m | 2Gi | 2000m |
| PredictionRUL | 2 | 2Gi | 1000m | 4Gi | 4000m |
| OrchestrateurMaintenance | 2 | 512Mi | 250m | 1Gi | 1000m |
| DashboardMonitoring | 2 | 512Mi | 250m | 1Gi | 1000m |

## 🔧 Health Checks

Tous les services ont :
- **Liveness Probe** : Vérifie que le service est vivant
- **Readiness Probe** : Vérifie que le service est prêt à recevoir du trafic

## 📝 Notes importantes

1. **Secrets** : Créer `secrets/secrets.yaml` depuis le template avant le déploiement
2. **Storage** : Adapter `storageClassName` dans PVC selon votre cluster
3. **Ingress** : Nécessite un Ingress Controller (nginx, traefik, etc.)
4. **Images** : Les images Docker doivent être buildées et pushées dans un registry
5. **Ressources** : Ajuster les limites selon votre cluster

## 🎯 Prochaines étapes (Phase 12)

- [ ] Documentation utilisateur finale
- [ ] Guide de déploiement production
- [ ] Monitoring et alertes Kubernetes
- [ ] Backup et restauration
- [ ] Scaling automatique (HPA)

## ✅ Phase 11 - TERMINÉE

Les configurations Kubernetes sont **complètes** et prêtes pour le déploiement !

