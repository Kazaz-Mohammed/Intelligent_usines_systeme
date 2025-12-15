# Guide Calibration et Quantification d'Incertitude

## Vue d'ensemble

Le service de calibration améliore la précision des prédictions RUL et quantifie l'incertitude pour fournir des intervalles de confiance fiables.

## Fonctionnalités

### 1. Méthodes de Calibration

#### Isotonic Regression ✅
- **Description** : Régression isotonique (monotone croissante)
- **Avantages** : Non-paramétrique, flexible
- **Utilisation** : Meilleur pour distributions non-linéaires

#### Platt Scaling ✅
- **Description** : Régression logistique pour mapper prédictions → valeurs réelles
- **Avantages** : Rapide, paramétrique
- **Utilisation** : Bon pour distributions proches de la normale

#### Temperature Scaling ✅
- **Description** : Optimisation d'un paramètre de température pour calibrer les prédictions
- **Avantages** : Simple, efficace pour modèles deep learning
- **Utilisation** : Idéal pour LSTM/GRU/TCN

### 2. Quantification d'Incertitude

#### Méthodes disponibles :
- **std** : Écart-type des prédictions d'ensemble
- **quantile** : Intervalles basés sur les percentiles (2.5%, 97.5%)
- **ensemble** : Agrégation des prédictions de plusieurs modèles

### 3. Intervalles de Confiance

- **Niveaux supportés** : 90%, 95%, 99%
- **Calcul automatique** : Basé sur l'incertitude et le niveau de confiance
- **Garantie** : RUL >= 0 (valeurs négatives clipées)

## Utilisation

### Configuration

Dans `app/config.py` :

```python
calibration_enabled: bool = True
calibration_method: str = "isotonic"  # isotonic, platt, temperature_scaling
```

### Entraîner la Calibration

```python
from app.services.calibration_service import CalibrationService

calibration_service = CalibrationService()

# Données de validation
predictions = np.array([100.0, 150.0, 200.0, ...])
actuals = np.array([95.0, 145.0, 195.0, ...])

# Entraîner
result = calibration_service.fit_calibration(
    predictions,
    actuals,
    method="isotonic"
)

print(result)
# {
#     "status": "success",
#     "method": "isotonic",
#     "metrics": {
#         "mae": 5.2,
#         "rmse": 7.1,
#         "mape": 3.5
#     }
# }
```

### Calibrer les Prédictions

```python
# Prédictions à calibrer
test_predictions = np.array([120.0, 180.0])

# Calibrer
calibrated = calibration_service.calibrate_predictions(test_predictions)
print(calibrated)  # [115.5, 175.2]
```

### Calculer l'Incertitude

```python
# Prédictions de plusieurs modèles (ensemble)
predictions_ensemble = np.array([
    [100.0, 150.0],  # LSTM
    [105.0, 155.0],  # GRU
    [95.0, 145.0]    # TCN
])

# Calculer l'incertitude
uncertainty, lower, upper = calibration_service.compute_uncertainty(
    predictions_ensemble,
    method="std"
)

print(f"Incertitude: {uncertainty}")
print(f"Intervalle: [{lower}, {upper}]")
```

### Calculer l'Intervalle de Confiance

```python
prediction = 150.0
uncertainty = 10.0

lower, upper = calibration_service.compute_confidence_interval(
    prediction,
    uncertainty,
    confidence_level=0.95
)

print(f"RUL: {prediction} ± {uncertainty}")
print(f"Intervalle 95%: [{lower}, {upper}]")
```

## API REST

### POST /api/v1/rul/calibrate

Calibre les modèles avec des données de validation.

**Request Body** :
```json
{
    "predictions": [100.0, 150.0, 200.0],
    "actuals": [95.0, 145.0, 195.0],
    "method": "isotonic"
}
```

**Response** :
```json
{
    "status": "success",
    "method": "isotonic",
    "metrics": {
        "mae": 5.2,
        "rmse": 7.1,
        "mape": 3.5
    }
}
```

### GET /api/v1/rul/status

Retourne le statut de la calibration.

**Response** :
```json
{
    "ready": true,
    "models": {...},
    "calibration": {
        "enabled": true,
        "method": "isotonic",
        "is_calibrated": true,
        "has_calibration_data": true
    }
}
```

## Intégration Automatique

La calibration est automatiquement appliquée lors des prédictions si :
1. `calibration_enabled = True` dans la config
2. Le service de calibration a été entraîné (`is_calibrated = True`)

**Exemple** :
```python
# Prédiction avec calibration automatique
result = service.predict_rul(request)

# result.rul_prediction est déjà calibré
# result.confidence_interval_lower/upper sont calculés avec incertitude
```

## Métriques de Calibration

- **MAE** (Mean Absolute Error) : Erreur moyenne absolue
- **RMSE** (Root Mean Squared Error) : Racine de l'erreur quadratique moyenne
- **MAPE** (Mean Absolute Percentage Error) : Erreur moyenne absolue en pourcentage

## Notes

1. **Données minimales** : Au moins 10 échantillons requis pour la calibration
2. **RUL >= 0** : Toutes les prédictions calibrées sont garanties >= 0
3. **Méthode par défaut** : Isotonic (configurable)
4. **Performance** : Temperature scaling est généralement plus rapide que isotonic

## Exemples

### Exemple 1 : Calibration avec Isotonic

```python
from app.services.calibration_service import CalibrationService
import numpy as np

service = CalibrationService()

# Données d'entraînement
predictions = np.random.rand(100) * 200
actuals = predictions * 0.9 + np.random.randn(100) * 5

# Entraîner
result = service.fit_calibration(predictions, actuals, method="isotonic")

# Utiliser
test_pred = np.array([150.0])
calibrated = service.calibrate_predictions(test_pred)
print(f"Original: {test_pred[0]}, Calibré: {calibrated[0]}")
```

### Exemple 2 : Incertitude d'Ensemble

```python
# Prédictions de 3 modèles
lstm_pred = 150.0
gru_pred = 155.0
tcn_pred = 145.0

predictions = np.array([[lstm_pred], [gru_pred], [tcn_pred]])

uncertainty, lower, upper = service.compute_uncertainty(
    predictions,
    method="std"
)

print(f"Moyenne: {np.mean([lstm_pred, gru_pred, tcn_pred])}")
print(f"Incertitude: {uncertainty[0]}")
print(f"Intervalle 95%: [{lower[0]}, {upper[0]}]")
```

## Troubleshooting

### Calibration non appliquée
- Vérifier `calibration_enabled = True`
- Vérifier `is_calibrated = True` dans le statut
- Vérifier que `fit_calibration()` a été appelé

### Erreur "insufficient_data"
- Fournir au moins 10 échantillons pour la calibration

### Prédictions négatives
- Les prédictions sont automatiquement clipées à 0 (RUL >= 0)

