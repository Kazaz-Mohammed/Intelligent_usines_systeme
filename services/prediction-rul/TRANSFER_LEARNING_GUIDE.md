# Guide Transfer Learning - NASA C-MAPSS

## Vue d'ensemble

Le service de transfer learning permet d'utiliser des modèles pré-entraînés sur le dataset NASA C-MAPSS pour améliorer les performances sur les données de l'usine.

## Fonctionnalités

### 1. Chargement de Modèles Pré-entraînés ✅
- Support pour LSTM, GRU, TCN
- Chargement depuis fichiers PyTorch (.pt)
- Vérification de compatibilité des poids

### 2. Application du Transfer Learning ✅
- Chargement automatique des poids pré-entraînés
- Filtrage des poids compatibles
- Option de gel des couches de l'encodeur

### 3. Fine-tuning ✅
- Fine-tuning avec learning rate réduit
- Support pour geler certaines couches
- Métriques de fine-tuning

### 4. Sauvegarde de Modèles ✅
- Sauvegarde de modèles pré-entraînés
- Format checkpoint PyTorch

## Utilisation

### Configuration

Dans `app/config.py` :

```python
transfer_learning_enabled: bool = True
transfer_learning_pretrained_path: Optional[str] = None  # Chemin par défaut
transfer_learning_freeze_layers: bool = False  # Geler les couches de l'encodeur
```

### Charger un Modèle Pré-entraîné

```python
from app.services.transfer_learning_service import TransferLearningService

service = TransferLearningService()

# Charger un modèle pré-entraîné
result = service.load_pretrained_model(
    "lstm",
    model_path="models/pretrained/lstm_cmapss.pt"
)

if result is not None:
    print("Modèle chargé avec succès")
```

### Appliquer le Transfer Learning

```python
from app.services.lstm_service import LSTMService, LSTMModel
import torch

# Créer un modèle
model = LSTMModel(input_size=5, hidden_size=64, num_layers=2)

# Appliquer transfer learning
transfer_service = TransferLearningService()
transfer_service.load_pretrained_model("lstm", model_path="model.pt")

model = transfer_service.apply_transfer_learning(
    model,
    "lstm",
    freeze_encoder=True  # Geler les couches LSTM
)
```

### Fine-tuning

```python
import numpy as np

# Données d'entraînement pour fine-tuning
X_train = np.random.randn(100, 10, 5)  # (samples, sequence, features)
y_train = np.random.rand(100) * 200

# Fine-tuning
result = transfer_service.fine_tune_model(
    model,
    X_train,
    y_train,
    epochs=10,
    learning_rate=0.0001,  # Learning rate plus faible
    batch_size=32
)

print(f"MAE: {result['mae']:.4f}, RMSE: {result['rmse']:.4f}")
```

### Sauvegarder un Modèle Pré-entraîné

```python
# Après entraînement sur NASA C-MAPSS
transfer_service.save_pretrained_model(
    model,
    "lstm",
    "models/pretrained/lstm_cmapss.pt",
    config={"input_size": 5, "hidden_size": 64}
)
```

## API REST

### POST /api/v1/rul/transfer-learning/load

Charge un modèle pré-entraîné.

**Request Body** :
```json
{
    "model_name": "lstm",
    "model_path": "models/pretrained/lstm_cmapss.pt"
}
```

**Response** :
```json
{
    "status": "success",
    "model_name": "lstm",
    "model_path": "models/pretrained/lstm_cmapss.pt",
    "message": "Modèle lstm chargé avec succès"
}
```

### GET /api/v1/rul/transfer-learning/info

Retourne des informations sur les modèles pré-entraînés chargés.

**Response** :
```json
{
    "enabled": true,
    "freeze_layers": false,
    "pretrained_models": {
        "lstm": {
            "path": "models/pretrained/lstm_cmapss.pt",
            "config": {
                "input_size": 5,
                "hidden_size": 64
            }
        }
    }
}
```

### GET /api/v1/rul/status

Le statut inclut maintenant les informations de transfer learning.

**Response** :
```json
{
    "ready": true,
    "models": {...},
    "calibration": {...},
    "transfer_learning": {
        "enabled": true,
        "freeze_layers": false,
        "pretrained_models": {...}
    }
}
```

## Intégration Automatique

Le transfer learning est automatiquement appliqué lors de l'entraînement si :
1. `transfer_learning_enabled = True` dans la config
2. Un modèle pré-entraîné est chargé pour le modèle correspondant

**Exemple** :
```python
# Le service charge automatiquement les modèles pré-entraînés
service = RULPredictionService()

# Lors de l'entraînement, le transfer learning est appliqué automatiquement
results = service.train_all_models(X_train, y_train)
```

## Format des Modèles Pré-entraînés

Les modèles doivent être sauvegardés au format PyTorch checkpoint :

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'model_name': 'lstm',
    'config': {
        'input_size': 5,
        'hidden_size': 64,
        'num_layers': 2
    }
}

torch.save(checkpoint, 'model.pt')
```

## Gel des Couches

### Pourquoi geler les couches ?
- **Avantages** : Préserve les connaissances apprises, réduit le risque d'overfitting
- **Inconvénients** : Moins de flexibilité pour s'adapter aux nouvelles données

### Quand geler ?
- **Geler** : Si les données de l'usine sont similaires à NASA C-MAPSS
- **Ne pas geler** : Si les données sont très différentes, permettre l'adaptation complète

## Exemples

### Exemple 1 : Transfer Learning Complet

```python
from app.services.transfer_learning_service import TransferLearningService
from app.services.lstm_service import LSTMService, LSTMModel
import numpy as np

# 1. Charger le modèle pré-entraîné
transfer_service = TransferLearningService()
transfer_service.load_pretrained_model("lstm", "models/pretrained/lstm_cmapss.pt")

# 2. Créer le service LSTM avec transfer learning
lstm_service = LSTMService(transfer_learning_service=transfer_service)

# 3. Entraîner (transfer learning appliqué automatiquement)
X_train = np.random.randn(100, 10, 5)
y_train = np.random.rand(100) * 200

metrics = lstm_service.train(X_train, y_train)
```

### Exemple 2 : Fine-tuning Manuel

```python
# Charger modèle pré-entraîné
transfer_service = TransferLearningService()
state_dict = transfer_service.load_pretrained_model("lstm", "model.pt")

# Créer modèle et appliquer poids
model = LSTMModel(input_size=5, hidden_size=64, num_layers=2)
model.load_state_dict(state_dict)

# Fine-tuning
X_train = np.random.randn(100, 10, 5)
y_train = np.random.rand(100) * 200

result = transfer_service.fine_tune_model(
    model,
    X_train,
    y_train,
    epochs=10,
    learning_rate=0.0001
)
```

## Notes

1. **Compatibilité** : Les poids doivent avoir la même forme que le modèle cible
2. **Learning Rate** : Utiliser un learning rate plus faible pour le fine-tuning (0.0001 vs 0.001)
3. **Epochs** : Moins d'epochs nécessaires avec transfer learning (10-20 vs 100+)
4. **Données** : Le transfer learning fonctionne mieux si les données sont similaires

## Troubleshooting

### Modèle non chargé
- Vérifier le chemin du fichier
- Vérifier que le fichier existe
- Vérifier le format du checkpoint

### Poids incompatibles
- Vérifier que les dimensions du modèle correspondent
- Vérifier la structure du state_dict

### Performance dégradée
- Essayer de ne pas geler les couches (`freeze_layers=False`)
- Augmenter le learning rate du fine-tuning
- Utiliser plus d'epochs

