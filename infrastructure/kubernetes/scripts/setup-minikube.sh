#!/bin/bash

# Script pour configurer minikube pour les tests

set -e

echo "🚀 Configuration de minikube pour Predictive Maintenance"
echo "========================================================"
echo ""

# Détecter le système d'exploitation
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    MINGW*)     MACHINE=Windows;;
    MSYS*)      MACHINE=Windows;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

# Vérifier si minikube est installé
if ! command -v minikube &> /dev/null; then
    echo "❌ minikube n'est pas installé"
    echo "Installer depuis: https://minikube.sigs.k8s.io/docs/start/"
    exit 1
fi

# Sur Windows, utiliser Hyper-V ou suggérer Docker Desktop
if [[ "$MACHINE" == "Windows" ]]; then
    echo "🖥️  Windows détecté"
    echo ""
    echo "⚠️  ATTENTION: Sur Windows, VirtualBox ne fonctionne pas avec Hyper-V"
    echo ""
    echo "💡 Options:"
    echo "   1. Utiliser Docker Desktop + Kind (recommandé):"
    echo "      ./scripts/setup-kind.sh"
    echo ""
    echo "   2. Utiliser Minikube avec Hyper-V:"
    echo "      ./scripts/setup-minikube-windows.sh"
    echo ""
    read -p "Continuer avec minikube (tentera d'utiliser le bon driver) ? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exécutez: ./scripts/setup-minikube-windows.sh ou ./scripts/setup-kind.sh"
        exit 0
    fi
fi

# Détecter le meilleur driver
if docker info &>/dev/null; then
    DRIVER="docker"
elif [[ "$MACHINE" == "Windows" ]]; then
    DRIVER="hyperv"
else
    DRIVER="virtualbox"
fi

echo "🔧 Utilisation du driver: $DRIVER"
echo ""

# Démarrer minikube
echo "1. Démarrage de minikube..."
if minikube status &>/dev/null; then
    echo "✅ minikube est déjà démarré"
else
    echo "⏳ Démarrage de minikube avec driver $DRIVER (cela peut prendre quelques minutes)..."
    if [[ "$DRIVER" == "hyperv" ]]; then
        minikube start --driver=hyperv --memory=8192 --cpus=4 --disk-size=20g
    elif [[ "$DRIVER" == "docker" ]]; then
        minikube start --driver=docker --memory=8192 --cpus=4
    else
        minikube start --driver=virtualbox --memory=8192 --cpus=4 --disk-size=20g
    fi
    echo "✅ minikube démarré"
fi

# Activer les addons nécessaires
echo ""
echo "2. Activation des addons minikube..."
minikube addons enable ingress
minikube addons enable metrics-server

# Configurer Docker pour utiliser minikube (si driver docker)
if [[ "$DRIVER" == "docker" ]]; then
    echo ""
    echo "3. Configuration de Docker pour minikube..."
    eval $(minikube docker-env)
else
    echo ""
    echo "3. Note: Pour builder les images, utilisez:"
    echo "   eval \$(minikube docker-env)"
fi

# Vérifier la configuration
echo ""
echo "4. Vérification de la configuration..."
kubectl cluster-info
kubectl get nodes

echo ""
echo "✅ minikube configuré!"
echo ""
echo "📝 Prochaines étapes:"
echo "1. Builder les images Docker:"
echo "   cd services/[service-name]"
echo "   docker build -t predictive-maintenance/[service-name]:latest ."
echo ""
echo "2. Déployer sur Kubernetes:"
echo "   cd infrastructure/kubernetes"
echo "   ./scripts/deploy-all.sh"
echo ""
echo "3. Tester le déploiement:"
echo "   ./scripts/test-deployment.sh"

