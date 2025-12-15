# Guide d'Installation Windows

## Problème: VirtualBox vs Hyper-V

Sur Windows, VirtualBox ne peut pas fonctionner quand Hyper-V est activé. Vous avez deux options :

### Option 1 : Utiliser Docker Desktop + Kind (RECOMMANDÉ)

C'est la solution la plus simple et la plus rapide sur Windows.

#### Prérequis
1. **Docker Desktop** installé et démarré
   - Télécharger: https://www.docker.com/products/docker-desktop
   - Activer WSL2 backend dans les paramètres

2. **Kind** installé
   ```powershell
   # Avec Chocolatey
   choco install kind
   
   # Ou avec Go
   go install sigs.k8s.io/kind@v0.20.0
   ```

#### Configuration
```bash
cd infrastructure/kubernetes
chmod +x scripts/*.sh
./scripts/setup-kind.sh
```

#### Avantages
- ✅ Pas de conflit avec Hyper-V
- ✅ Plus rapide que minikube
- ✅ Utilise Docker directement
- ✅ Pas besoin de VM

---

### Option 2 : Utiliser Minikube avec Hyper-V

Si vous préférez utiliser minikube, vous devez utiliser le driver Hyper-V.

#### Prérequis
1. **Hyper-V activé** sur Windows
   - Vérifier: `Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V`
   - Activer si nécessaire (nécessite redémarrage)

2. **Minikube** installé
   ```powershell
   choco install minikube
   ```

#### Configuration
```bash
cd infrastructure/kubernetes
chmod +x scripts/*.sh
./scripts/setup-minikube-windows.sh
```

Ou manuellement:
```bash
minikube start --driver=hyperv --memory=8192 --cpus=4
minikube addons enable ingress
minikube addons enable metrics-server
```

---

### Option 3 : Désactiver Hyper-V (NON RECOMMANDÉ)

⚠️ **Attention**: Désactiver Hyper-V peut affecter d'autres applications (Docker Desktop, WSL2, etc.)

Si vous voulez vraiment utiliser VirtualBox:

1. Ouvrir PowerShell en **administrateur**
2. Désactiver Hyper-V:
   ```powershell
   bcdedit /set hypervisorlaunchtype off
   ```
3. **Redémarrer** l'ordinateur
4. Démarrer minikube:
   ```bash
   minikube start --driver=virtualbox --memory=8192 --cpus=4
   ```

Pour réactiver Hyper-V plus tard:
```powershell
bcdedit /set hypervisorlaunchtype auto
# Redémarrer
```

---

## Comparaison des options

| Option | Facilité | Performance | Compatibilité |
|--------|----------|-------------|---------------|
| **Docker Desktop + Kind** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Minikube + Hyper-V** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Minikube + VirtualBox** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

## Recommandation

**Utilisez Docker Desktop + Kind** pour Windows. C'est la solution la plus simple et la plus compatible.

## Workflow complet avec Kind

```bash
# 1. Vérifier Docker Desktop
docker info

# 2. Configurer kind
cd infrastructure/kubernetes
./scripts/setup-kind.sh

# 3. Builder les images
./scripts/build-all-images.sh

# 4. Charger les images dans kind
kind load docker-image predictive-maintenance/ingestion-iiot:latest --name predictive-maintenance
kind load docker-image predictive-maintenance/preprocessing:latest --name predictive-maintenance
kind load docker-image predictive-maintenance/extraction-features:latest --name predictive-maintenance
kind load docker-image predictive-maintenance/detection-anomalies:latest --name predictive-maintenance
kind load docker-image predictive-maintenance/prediction-rul:latest --name predictive-maintenance
kind load docker-image predictive-maintenance/orchestrateur-maintenance:latest --name predictive-maintenance
kind load docker-image predictive-maintenance/dashboard-monitoring:latest --name predictive-maintenance

# 5. Déployer
./scripts/deploy-all.sh

# 6. Tester
./scripts/test-deployment.sh
```

## Dépannage

### Problème: "Docker Desktop is not running"
- Démarrer Docker Desktop
- Vérifier que WSL2 backend est activé

### Problème: "kind: command not found"
- Installer kind: `choco install kind`
- Vérifier PATH: `where kind`

### Problème: "Hyper-V is not enabled"
- Activer Hyper-V dans "Activer ou désactiver des fonctionnalités Windows"
- Redémarrer

### Problème: "VirtualBox won't boot"
- Utiliser Hyper-V ou Docker Desktop + Kind
- Ne pas désactiver Hyper-V (affecte Docker Desktop)

