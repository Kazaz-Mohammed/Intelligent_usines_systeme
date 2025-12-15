# Guide MLflow

## Description

MLflow est intégré dans le service de détection d'anomalies pour :
- **Tracking** : Suivre les expériences d'entraînement
- **Registry** : Enregistrer et versionner les modèles
- **Logging** : Logger les paramètres, métriques et artefacts

## Configuration

### Variables d'environnement

```bash
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=anomaly-detection
MLFLOW_ENABLED=true
```

### Désactiver MLflow

Pour désactiver MLflow (par exemple dans les tests) :

```bash
MLFLOW_ENABLED=false
```

Quand MLflow est désactivé, toutes les méthodes retournent silencieusement sans effet.

## Démarrer MLflow

### Serveur MLflow local

```bash
# Démarrer le serveur MLflow
mlflow server --host 0.0.0.0 --port 5000

# Interface UI disponible sur
# http://localhost:5000
```

### Avec Docker

```bash
docker run -p 5000:5000 \
  -e MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db \
  -e MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlruns \
  ghcr.io/mlflow/mlflow:v2.9.2
```

## Utilisation

### Entraîner un modèle (MLflow automatique)

Quand vous entraînez un modèle via l'API ou directement, MLflow est automatiquement utilisé :

```python
from app.services.isolation_forest_service import IsolationForestService
import numpy as np

service = IsolationForestService()
X = np.random.randn(100, 5)  # Données d'entraînement

# L'entraînement log automatiquement dans MLflow
metrics = service.train(X, feature_names=["feature1", "feature2", ...])
```

MLflow log automatiquement :
- **Paramètres** : contamination, n_estimators, n_samples, n_features
- **Métriques** : anomaly_rate, mean_score, std_score
- **Modèle** : Modèle sklearn enregistré dans le registry

### Voir les expériences

1. Ouvrir l'interface MLflow : `http://localhost:5000`
2. Sélectionner l'experiment `anomaly-detection`
3. Voir tous les runs d'entraînement

### Voir les métriques

Dans l'interface MLflow, vous pouvez voir :

- **Métriques par modèle** :
  - `n_anomalies_detected` : Nombre d'anomalies détectées dans les données d'entraînement
  - `anomaly_rate` : Taux d'anomalies (contamination effective)
  - `mean_score` : Score moyen
  - `std_score` : Écart-type des scores

- **Métriques LSTM Autoencoder** :
  - `train_loss` : Loss d'entraînement (par epoch)
  - `final_loss` : Loss finale
  - `mean_reconstruction_error` : Erreur de reconstruction moyenne
  - `threshold` : Seuil utilisé pour la détection

### Charger un modèle depuis le registry

```python
from app.services.mlflow_service import MLflowService

mlflow_service = MLflowService()

# Charger le dernier modèle Isolation Forest en production
model = mlflow_service.load_model("isolation_forest", stage="Production")

# Charger une version spécifique
model = mlflow_service.load_model("isolation_forest", version=1)

# Charger le dernier modèle en staging
model = mlflow_service.load_model("isolation_forest", stage="Staging")
```

### Transitionner un modèle vers Production

```python
from app.services.mlflow_service import MLflowService

mlflow_service = MLflowService()

# Transitionner la version 2 du modèle vers Production
mlflow_service.transition_model_stage(
    model_name="isolation_forest",
    version=2,
    stage="Production"
)
```

## Modèles enregistrés

### Isolation Forest

- **Registry name** : `isolation_forest`
- **Type** : sklearn (PyOD)
- **Paramètres loggés** :
  - `contamination` : Taux de contamination attendu
  - `n_estimators` : Nombre d'arbres
  - `max_samples` : Nombre max d'échantillons par arbre
- **Métriques loggées** :
  - `n_anomalies_detected`
  - `anomaly_rate`
  - `mean_score`, `std_score`

### One-Class SVM

- **Registry name** : `one_class_svm`
- **Type** : sklearn (PyOD)
- **Paramètres loggés** :
  - `nu` : Paramètre nu (upper bound sur l'erreur d'entraînement)
  - `kernel` : Type de kernel (rbf, linear, poly, sigmoid)
  - `gamma` : Paramètre gamma
- **Métriques loggées** :
  - `n_anomalies_detected`
  - `anomaly_rate`
  - `mean_score`, `std_score`

### LSTM Autoencoder

- **Registry name** : `lstm_autoencoder`
- **Type** : PyTorch
- **Paramètres loggés** :
  - `encoder_layers` : Architecture de l'encodeur
  - `decoder_layers` : Architecture du décodeur
  - `sequence_length` : Longueur des séquences
  - `batch_size` : Taille du batch
  - `epochs` : Nombre d'epochs
  - `learning_rate` : Taux d'apprentissage
- **Métriques loggées** :
  - `train_loss` : Par epoch
  - `final_loss` : Loss finale
  - `mean_reconstruction_error` : Erreur de reconstruction moyenne
  - `threshold` : Seuil de détection

## Structure des runs MLflow

Chaque run d'entraînement contient :

```
run_<uuid>/
├── artifacts/
│   └── <model_name>_model/
│       ├── model.pkl (ou model.pth pour PyTorch)
│       ├── conda.yaml
│       ├── requirements.txt
│       └── MLmodel
├── metrics/
│   ├── anomaly_rate
│   ├── mean_score
│   └── ...
├── params/
│   ├── contamination
│   ├── n_estimators
│   └── ...
└── tags/
```

## API MLflow

### Récupérer les métriques via l'API REST

```bash
# Récupérer les dernières versions des modèles
curl http://localhost:8084/api/v1/anomalies/metrics
```

(Actuellement placeholder, à implémenter)

### Utiliser l'API MLflow directement

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5000")

# Lister les expériences
experiments = client.list_experiments()
print([exp.name for exp in experiments])

# Récupérer les runs d'une expérience
runs = client.search_runs(
    experiment_ids=["0"],  # ID de l'experiment
    max_results=10
)

# Récupérer un run spécifique
run = client.get_run(run_id="<run_id>")
print(f"Metrics: {run.data.metrics}")
print(f"Params: {run.data.params}")

# Récupérer les versions d'un modèle
versions = client.get_latest_versions("isolation_forest", stages=["Production"])
print([v.version for v in versions])
```

## Bonnes pratiques

### 1. Nommer les runs

Les runs sont automatiquement nommés avec `{model_name}_{n_samples}_samples`. Pour un nom personnalisé :

```python
from app.services.mlflow_service import MLflowService

mlflow_service = MLflowService()
run = mlflow_service.start_run(run_name="experiment_2024_01_01")
# ... entraînement ...
mlflow_service.end_run()
```

### 2. Versionner les modèles

Après chaque entraînement, les modèles sont automatiquement enregistrés dans le registry. Pour les versionner :

1. Voir les versions dans l'UI MLflow
2. Transitionner vers "Staging" pour tester
3. Transitionner vers "Production" une fois validé

### 3. Comparer les runs

Dans l'UI MLflow :
- Sélectionner plusieurs runs
- Comparer les métriques
- Voir les différences de paramètres

### 4. Sauvegarder les métriques custom

Pour ajouter des métriques custom :

```python
from app.services.mlflow_service import MLflowService

mlflow_service = MLflowService()
mlflow_service.log_metrics({
    "custom_metric_1": 0.95,
    "custom_metric_2": 0.87
})
```

## Troubleshooting

### MLflow ne log pas

1. **Vérifier que MLflow est activé** :
   ```bash
   echo $MLFLOW_ENABLED  # Doit être "true"
   ```

2. **Vérifier la connexion au serveur MLflow** :
   ```bash
   curl http://localhost:5000/health
   ```

3. **Vérifier les logs** : Le service log des erreurs si MLflow ne peut pas être initialisé

### Modèles non trouvés dans le registry

1. **Vérifier que le modèle a été enregistré** : Voir dans l'UI MLflow
2. **Vérifier le nom du modèle** : Doit correspondre exactement
3. **Vérifier les stages** : Le modèle peut être dans "None" stage

### Problèmes de performance

Si MLflow ralentit l'entraînement :

1. **Désactiver temporairement** : `MLFLOW_ENABLED=false`
2. **Utiliser un backend store plus rapide** : PostgreSQL au lieu de SQLite
3. **Configurer un artifact store distant** : S3, Azure Blob, etc.

## Exemples complets

Voir les fichiers de tests pour des exemples complets :
- `tests/test_isolation_forest_service.py`
- `tests/test_one_class_svm_service.py`
- `tests/test_lstm_autoencoder_service.py`

