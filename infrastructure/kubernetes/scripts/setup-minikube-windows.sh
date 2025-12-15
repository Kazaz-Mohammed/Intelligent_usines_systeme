#!/bin/bash

# Script pour configurer minikube sur Windows (avec Hyper-V)

set -e

echo "🚀 Configuration de minikube pour Predictive Maintenance (Windows)"
echo "=================================================================="
echo ""

# Vérifier si minikube est installé
if ! command -v minikube &> /dev/null; then
    echo "❌ minikube n'est pas installé"
    echo "Installer depuis: https://minikube.sigs.k8s.io/docs/start/"
    exit 1
fi

# Détecter le système d'exploitation
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    MINGW*)     MACHINE=Windows;;
    MSYS*)      MACHINE=Windows;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "🖥️  Système détecté: $MACHINE"
echo ""

# Sur Windows, utiliser Hyper-V ou Docker
if [[ "$MACHINE" == "Windows" ]]; then
    echo "📋 Windows détecté - Configuration pour Hyper-V ou Docker"
    echo ""
    
    # Vérifier si Docker Desktop est disponible
    if docker info &>/dev/null; then
        echo "✅ Docker Desktop détecté"
        echo ""
        echo "💡 Recommandation: Utiliser Docker Desktop avec kind (plus simple)"
        echo "   Exécuter: ./scripts/setup-kind.sh"
        echo ""
        read -p "Continuer avec minikube (Hyper-V) ? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Exécutez: ./scripts/setup-kind.sh"
            exit 0
        fi
    fi
    
    # Utiliser le driver Hyper-V
    DRIVER="hyperv"
    echo "🔧 Utilisation du driver: $DRIVER"
else
    # Sur Linux/Mac, utiliser docker ou virtualbox
    if docker info &>/dev/null; then
        DRIVER="docker"
    else
        DRIVER="virtualbox"
    fi
    echo "🔧 Utilisation du driver: $DRIVER"
fi

# Démarrer minikube
echo ""
echo "1. Démarrage de minikube avec driver $DRIVER..."
if minikube status &>/dev/null; then
    echo "✅ minikube est déjà démarré"
    CURRENT_DRIVER=$(minikube profile list -o json | jq -r '.valid[0].Config.Driver' 2>/dev/null || echo "unknown")
    if [[ "$CURRENT_DRIVER" != "$DRIVER" ]]; then
        echo "⚠️  Driver actuel: $CURRENT_DRIVER, nouveau driver: $DRIVER"
        echo "⏳ Arrêt de minikube pour changer de driver..."
        minikube stop
        minikube delete
    fi
fi

if ! minikube status &>/dev/null; then
    echo "⏳ Démarrage de minikube (cela peut prendre quelques minutes)..."
    
    if [[ "$DRIVER" == "hyperv" ]]; then
        # Sur Windows avec Hyper-V
        minikube start --driver=hyperv --memory=8192 --cpus=4 --disk-size=20g
    elif [[ "$DRIVER" == "docker" ]]; then
        # Avec Docker
        minikube start --driver=docker --memory=8192 --cpus=4
    else
        # VirtualBox (Linux/Mac)
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
    echo "   # Puis builder vos images Docker"
fi

# Vérifier la configuration
echo ""
echo "4. Vérification de la configuration..."
kubectl cluster-info
kubectl get nodes

echo ""
echo "✅ minikube configuré avec le driver $DRIVER!"
echo ""
echo "📝 Prochaines étapes:"
echo "1. Builder les images Docker:"
if [[ "$DRIVER" == "docker" ]]; then
    echo "   # Docker est déjà configuré"
else
    echo "   eval \$(minikube docker-env)"
fi
echo "   cd services/[service-name]"
echo "   docker build -t predictive-maintenance/[service-name]:latest ."
echo ""
echo "2. Déployer sur Kubernetes:"
echo "   cd infrastructure/kubernetes"
echo "   ./scripts/deploy-all.sh"
echo ""
echo "3. Tester le déploiement:"
echo "   ./scripts/test-deployment.sh"

