#!/bin/bash

# Script pour configurer kind (Kubernetes in Docker) pour les tests

set -e

echo "🚀 Configuration de kind pour Predictive Maintenance"
echo "====================================================="
echo ""

# Vérifier si kind est installé
if ! command -v kind &> /dev/null; then
    echo "❌ kind n'est pas installé"
    echo "Installer depuis: https://kind.sigs.k8s.io/docs/user/quick-start/"
    exit 1
fi

# Vérifier si Docker est en cours d'exécution
if ! docker info &>/dev/null; then
    echo "❌ Docker n'est pas en cours d'exécution"
    exit 1
fi

# Créer le cluster kind
CLUSTER_NAME="predictive-maintenance"

echo "1. Vérification du cluster kind..."
if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
    echo "✅ Cluster $CLUSTER_NAME existe déjà"
    kind get kubeconfig --name $CLUSTER_NAME > ~/.kube/kind-config
else
    echo "⏳ Création du cluster kind (cela peut prendre quelques minutes)..."
    cat <<EOF | kind create cluster --name $CLUSTER_NAME --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
  - containerPort: 443
    hostPort: 443
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
EOF
    echo "✅ Cluster créé"
fi

# Configurer kubectl
echo ""
echo "2. Configuration de kubectl..."
kind get kubeconfig --name $CLUSTER_NAME > ~/.kube/kind-config
export KUBECONFIG=~/.kube/kind-config

# Installer Ingress Controller
echo ""
echo "3. Installation de l'Ingress Controller..."
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=300s

# Vérifier la configuration
echo ""
echo "4. Vérification de la configuration..."
kubectl cluster-info
kubectl get nodes

echo ""
echo "✅ kind configuré!"
echo ""
echo "📝 Prochaines étapes:"
echo "1. Charger les images Docker dans kind:"
echo "   kind load docker-image predictive-maintenance/[service-name]:latest --name $CLUSTER_NAME"
echo ""
echo "2. Déployer sur Kubernetes:"
echo "   cd infrastructure/kubernetes"
echo "   ./scripts/deploy-all.sh"
echo ""
echo "3. Tester le déploiement:"
echo "   ./scripts/test-deployment.sh"

