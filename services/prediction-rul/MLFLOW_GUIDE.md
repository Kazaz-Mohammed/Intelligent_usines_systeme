# Guide MLflow - Service Prediction RUL

## Vue d'ensemble

Le service MLflow permet de tracker les expériences d'entraînement, comparer les modèles, et gérer le registre des modèles RUL.

## Fonctionnalités

### 1. Tracking des Expériences ✅
- Logging automatique des paramètres d'entraînement
- Logging des métriques (MAE, RMSE, R², Loss)
- Logging des modèles PyTorch et scikit-learn
- Comparaison des runs

### 2. Registre des Modèles ✅
- Enregistrement automatique des modèles
- Gestion des versions
- Stages (Staging, Production, Archived)
- Chargement de modèles depuis le registry

### 3. Intégration Automatique ✅
- Intégré dans tous les services de modèles (LSTM, GRU, TCN, XGBoost)
- Logging automatique lors de l'entraînement
- Pas de code supplémentaire requis

## Configuration

Dans `app/config.py` ou `.env` :

```python
mlflow_enabled: bool = True
mlflow_tracking_uri: str = "http://localhost:5000"
mlflow_experiment_name: str = "rul-prediction"
```

## Utilisation

### Démarrer MLflow

```bash
# Démarrer le serveur MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Ou avec tracking URI personnalisé
mlflow ui --backend-store-uri http://localhost:5000
```

Accéder à l'interface : http://localhost:5000

### Entraînement avec MLflow

L'entraînement log automatiquement dans MLflow :

```python
from app.services.rul_prediction_service import RULPredictionService
import numpy as np

service = RULPredictionService()

# Entraîner les modèles (MLflow logging automatique)
results = service.train_all_models(X_train, y_train)

# Les runs sont automatiquement créées dans MLflow
```

### Utiliser le Service MLflow Directement

```python
from app.services.mlflow_service import MLflowService

mlflow_service = MLflowService()

# Démarrer une run
run = mlflow_service.start_run(run_name="Custom_Training")

# Logger des paramètres
mlflow_service.log_params({
    "epochs": 100,
    "learning_rate": 0.001,
    "batch_size": 32
})

# Logger des métriques
mlflow_service.log_metrics({
    "mae": 5.0,
    "rmse": 7.0,
    "r2": 0.95
})

# Logger un modèle
mlflow_service.log_model(
    model,
    "model_path",
    registered_model_name="LSTM_RUL_Model"
)

# Terminer la run
mlflow_service.end_run()
```

### Charger un Modèle depuis MLflow

```python
from app.services.mlflow_service import MLflowService

mlflow_service = MLflowService()

# Charger par version
model = mlflow_service.load_model(
    "models:/LSTM_RUL_Model/1",
    model_type="pytorch"
)

# Charger par stage
model = mlflow_service.load_model(
    "models:/LSTM_RUL_Model/Production",
    model_type="pytorch"
)
```

### Rechercher des Runs

```python
# Rechercher les meilleures runs
runs = mlflow_service.search_runs(
    filter_string="metrics.mae < 5.0",
    max_results=10
)

for run in runs:
    print(f"Run: {run['run_id']}, MAE: {run['metrics'].get('mae', 'N/A')}")
```

## API REST

### GET /api/v1/rul/status

Le statut inclut maintenant les informations MLflow :

```json
{
    "ready": true,
    "models": {...},
    "calibration": {...},
    "transfer_learning": {...},
    "mlflow": {
        "enabled": true,
        "tracking_uri": "http://localhost:5000",
        "experiment_name": "rul-prediction"
    }
}
```

## Modèles Enregistrés

Les modèles sont automatiquement enregistrés avec ces noms :

- **LSTM_RUL_Model** - Modèle LSTM
- **GRU_RUL_Model** - Modèle GRU
- **TCN_RUL_Model** - Modèle TCN
- **XGBoost_RUL_Model** - Modèle XGBoost

## Métriques Trackées

### Métriques d'Entraînement
- `train_mae` - Mean Absolute Error (entraînement)
- `train_rmse` - Root Mean Squared Error (entraînement)
- `train_r2` - Coefficient of Determination (entraînement)
- `final_train_loss` - Loss finale d'entraînement

### Métriques de Validation (si données fournies)
- `val_mae` - Mean Absolute Error (validation)
- `val_rmse` - Root Mean Squared Error (validation)
- `val_r2` - Coefficient of Determination (validation)
- `best_val_loss` - Meilleure loss de validation

## Paramètres Trackés

Tous les paramètres d'entraînement sont trackés :
- Architecture du modèle (hidden_size, num_layers, etc.)
- Hyperparamètres (learning_rate, batch_size, epochs)
- Configuration (device, sequence_length, etc.)

## Interface MLflow UI

### Accéder à l'Interface

1. Démarrer MLflow :
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

2. Ouvrir dans le navigateur : http://localhost:5000

### Fonctionnalités de l'UI

- **Runs** : Voir toutes les runs d'entraînement
- **Comparaison** : Comparer plusieurs runs
- **Métriques** : Visualiser les métriques au cours du temps
- **Modèles** : Voir les modèles enregistrés
- **Registry** : Gérer les versions et stages

## Exemples

### Exemple 1 : Entraînement avec MLflow

```python
from app.services.rul_prediction_service import RULPredictionService
import numpy as np

service = RULPredictionService()

# Données d'entraînement
X_train = np.random.randn(1000, 10, 5)
y_train = np.random.rand(1000) * 200

# Entraîner (MLflow logging automatique)
results = service.train_all_models(X_train, y_train)

# Vérifier dans MLflow UI
# http://localhost:5000
```

### Exemple 2 : Charger le Meilleur Modèle

```python
from app.services.mlflow_service import MLflowService

mlflow_service = MLflowService()

# Rechercher la meilleure run
runs = mlflow_service.search_runs(
    filter_string="metrics.val_mae < 5.0",
    max_results=1
)

if runs:
    best_run = runs[0]
    print(f"Meilleure run: {best_run['run_id']}, MAE: {best_run['metrics'].get('val_mae')}")
    
    # Charger le modèle
    model_uri = f"runs:/{best_run['run_id']}/lstm_model"
    model = mlflow_service.load_model(model_uri, model_type="pytorch")
```

### Exemple 3 : Comparer les Modèles

```python
# Rechercher toutes les runs LSTM
lstm_runs = mlflow_service.search_runs(
    filter_string="params.model_name = 'lstm'",
    max_results=10
)

# Trier par MAE
lstm_runs.sort(key=lambda x: x['metrics'].get('val_mae', float('inf')))

print("Top 3 LSTM models:")
for i, run in enumerate(lstm_runs[:3]):
    print(f"{i+1}. Run {run['run_id']}: MAE={run['metrics'].get('val_mae')}")
```

## Troubleshooting

### MLflow non accessible
- Vérifier que le serveur MLflow est démarré
- Vérifier `mlflow_tracking_uri` dans la config
- Vérifier les logs pour les erreurs de connexion

### Modèles non enregistrés
- Vérifier `mlflow_enabled = True`
- Vérifier les logs pour les erreurs
- Vérifier les permissions d'écriture

### Erreurs de chargement
- Vérifier que le modèle existe dans le registry
- Vérifier la version/stage du modèle
- Vérifier le type de modèle (pytorch vs sklearn)

## Notes

1. **Performance** : MLflow peut ralentir légèrement l'entraînement
2. **Stockage** : Les modèles sont stockés localement par défaut
3. **Backend** : Utiliser une base de données pour la production (PostgreSQL, MySQL)
4. **Artifacts** : Les modèles sont stockés comme artifacts MLflow

## Production

Pour la production, configurer :

```python
# Backend store (PostgreSQL)
mlflow_tracking_uri = "postgresql://user:password@localhost/mlflow"

# Artifact store (S3, Azure, etc.)
# Configuré via MLflow UI ou variables d'environnement
```

