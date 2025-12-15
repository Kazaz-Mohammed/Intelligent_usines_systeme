# 04_ml_data_preparation.knwf – Design du workflow KNIME

## Objectif

Préparer les jeux de données finaux pour l’entraînement des modèles ML/DL :
- Jointure features + RUL.
- Normalisation/standardisation.
- Création de splits train/validation/test.
- (Optionnel) création de séquences pour modèles temporels (LSTM/GRU).

## Entrées

- Table de features sélectionnées :  
  - `data-mining/outputs/feature-selection/train_FD001_selected_features.csv`
- Table RUL correspondante :  
  - `datasets/nasa-cmapss/RUL_FD001.txt`

## Étapes KNIME

1. **CSV Reader – Features**  
   - Lire `train_FD001_selected_features.csv`.

2. **File Reader – RUL**  
   - Lire `RUL_FD001.txt` (une valeur RUL par moteur dans l’ordre des IDs).  
   - Ajouter une colonne `id` si nécessaire (1, 2, 3, …).

3. **Joiner**  
   - Joindre features + RUL par `id` (ou `(id, cycle)` selon la définition que tu choisis).

4. **Missing Value**  
   - Node : `Missing Value`  
   - Traiter les éventuelles valeurs manquantes (remplacement par moyenne/médiane, ou suppression de lignes).

5. **Normalizer / Standardizer**  
   - Node : `Normalizer` ou `Numeric Binner`, etc.  
   - Appliquer une normalisation cohérente avec la Phase 4 (z-score, min-max…).

6. **Partitioning**  
   - Node : `Partitioning`  
   - Split en :
     - Train (ex : 70%)
     - Validation (ex : 15%)
     - Test (ex : 15%)
   - Variante : d’abord train/test, puis partitionner train en train/val.

7. **(Optionnel) Lag Column / Moving Window**  
   - Pour préparer des séquences pour LSTM/GRU :
     - Utiliser des nodes comme `Lag Column`, `Moving Aggregation`, ou des extensions KNIME pour séries temporelles.

8. **CSV Writer**  
   - Exporter les tables finales :
     - `data-mining/outputs/ml/train.csv`
     - `data-mining/outputs/ml/val.csv`
     - `data-mining/outputs/ml/test.csv`

## Création du `.knwf`

1. Créer un workflow KNIME nommé `04_ml_data_preparation`.
2. Construire la chaîne de nodes décrite ci-dessus.
3. Sauvegarder le workflow dans `data-mining/knime-workflows/` pour obtenir `04_ml_data_preparation.knwf`.


