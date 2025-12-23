# ML Pipeline - Maintenance Prédictive

Ce dossier contient le pipeline d'entraînement des modèles ML pour la maintenance prédictive.

## 📋 Contenu

- **ml_pipeline_tutorial.ipynb** : Notebook Jupyter complet avec le pipeline ML
- **saved_models/** : Modèles entraînés sauvegardés
  - `lstm_rul_model.pth` : Modèle LSTM pour prédiction RUL
  - `xgboost_rul_model.pkl` : Modèle XGBoost pour prédiction RUL
  - `isolation_forest_model.pkl` : Modèle Isolation Forest pour détection d'anomalies
  - `feature_scaler.pkl` : Scaler pour normalisation des features

## 🎯 Objectifs

Ce pipeline ML implémente :

1. **Prédiction RUL (Remaining Useful Life)**
   - Modèle LSTM (Long Short-Term Memory)
   - Modèle XGBoost
   - Évaluation avec métriques MAE, RMSE, R²

2. **Détection d'Anomalies**
   - Isolation Forest
   - Autoencodeur LSTM
   - Détection temps-réel

3. **Extraction de Features**
   - Features temporelles (moyenne, écart-type, min, max)
   - Features fréquentielles (FFT, spectre de puissance)
   - Normalisation et scaling

## 🚀 Utilisation

### Prérequis

```bash
# Installer les dépendances Python
pip install numpy pandas matplotlib seaborn scikit-learn torch xgboost pyod scipy optuna mlflow tqdm jupyter
```

### Exécution

1. **Ouvrir le notebook**
```bash
jupyter notebook ml_pipeline_tutorial.ipynb
```

2. **Exécuter les cellules dans l'ordre**
   - Le notebook charge les données NASA C-MAPSS
   - Prétraite les données
   - Extrait les features
   - Entraîne les modèles
   - Évalue les performances
   - Sauvegarde les modèles dans `saved_models/`

3. **Utiliser les modèles entraînés**
   - Les modèles sauvegardés sont utilisés par les services `detection-anomalies` et `prediction-rul`
   - Les services chargent automatiquement les modèles depuis `saved_models/`

## 📊 Dataset

**NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation)
- Localisation : `../datasets/nasa-cmapss/`
- Format : CSV
- 21 capteurs + 3 réglages moteur

## 🔄 Intégration avec les Services

Les modèles entraînés sont utilisés par :

- **Service Détection-Anomalies** (`services/detection-anomalies/`)
  - Charge `isolation_forest_model.pkl`
  - Utilise `feature_scaler.pkl` pour normalisation

- **Service Prédiction-RUL** (`services/prediction-rul/`)
  - Charge `lstm_rul_model.pth` et `xgboost_rul_model.pkl`
  - Utilise `feature_scaler.pkl` pour normalisation

## 📈 Métriques de Performance

### Prédiction RUL
- **MAE** (Mean Absolute Error) : ~15-20 cycles
- **RMSE** (Root Mean Squared Error) : ~20-25 cycles
- **R² Score** : ~0.85-0.90

### Détection d'Anomalies
- **Précision** : ~0.90-0.95
- **Rappel** : ~0.85-0.90
- **F1-Score** : ~0.87-0.92

## 🔧 Réentraînement

Pour réentraîner les modèles avec de nouvelles données :

```bash
# Utiliser les scripts Python
python scripts/train_models.py
python scripts/train_anomaly_with_real_features.py
python scripts/retrain_anomaly_models.py
```

## 📝 Notes

- Les modèles sont sauvegardés au format `.pkl` (scikit-learn) et `.pth` (PyTorch)
- Le scaler doit être réentraîné si les features changent
- Les modèles doivent être compatibles avec les versions de bibliothèques utilisées dans les services

