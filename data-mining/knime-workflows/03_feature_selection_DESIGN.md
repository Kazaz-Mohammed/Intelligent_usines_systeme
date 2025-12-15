# 03_feature_selection.knwf – Design du workflow KNIME

## Objectif

Sélectionner un sous-ensemble de features pertinentes pour :
- La **détection d’anomalies** (Phase 6).
- La **prédiction de RUL** (Phase 7).

## Entrées

- Table de features/statistiques issue de `02_statistical_feature_analysis.knwf` :  
  - `data-mining/outputs/statistics/statistics_train_FD001_sensors.csv`
- Optionnel : table contenant RUL ou labels dérivés, par exemple à partir de `RUL_FD001.txt`.

## Étapes KNIME

1. **File Reader / CSV Reader**  
   - Lire la table de statistiques/features.

2. **Joiner (optionnel si RUL disponible)**  
   - Joindre la table de features avec la table RUL par `id` ou `(id, cycle)`.

3. **Correlation Filter**  
   - Node : `Column Filter` ou `Correlation Filter` custom.  
   - Supprimer les features fortement corrélées entre elles selon un seuil (ex : |corr| > 0.95).

4. **Feature Importance (optionnel)**  
   - Utiliser un modèle simple (ex : Random Forest, Gradient Boosted Trees via KNIME) :
     - Node : `Random Forest Learner`
     - Node : `Random Forest Predictor`
   - Extraire l’importance des variables avec `Feature Importance` / `Permutation Feature Importance` si disponible.

5. **Ranker**  
   - Classer les features selon leur importance (ou une métrique choisie).

6. **Column Filter**  
   - Garder uniquement les N meilleures features (ex : top 20).

## Sorties recommandées

- Liste des features sélectionnées (table avec nom + score d’importance).
- Table réduite avec uniquement les features sélectionnées.
- Exports :
  - `data-mining/outputs/feature-selection/selected_features_FD001.csv`
  - `data-mining/outputs/feature-selection/train_FD001_selected_features.csv`

## Création du `.knwf`

1. Créer un workflow KNIME nommé `03_feature_selection`.
2. Ajouter les nodes dans l’ordre décrit.
3. Sauvegarder dans `data-mining/knime-workflows/` pour obtenir `03_feature_selection.knwf`.


