 # Phase 5 – Data Mining avec KNIME (Plan des workflows)

 ## Objectif général

 Mettre en place une **chaîne de data mining KNIME** pour :
 - Explorer le dataset **NASA C-MAPSS**.
 - Analyser les **features** (brutes et extraites) des moteurs.
 - Sélectionner les variables pertinentes pour les modèles ML/DL (anomalies et RUL).
 - Produire des **datasets préparés** qui seront utilisés dans les phases 6 et 7.

 Le dataset est déjà téléchargé dans :
 - `datasets/nasa-cmapss/`

 Fichiers principaux :
 - `train_FD001.txt` … `train_FD004.txt`
 - `test_FD001.txt` … `test_FD004.txt`
 - `RUL_FD001.txt` … `RUL_FD004.txt`

 ---

 ## Structure recommandée des workflows KNIME

 Tous les workflows seront stockés dans ce dossier :
 - `data-mining/knime-workflows/`

 Workflows prévus (fichiers `.knwf`) :

 1. **01_exploration_cmapss.knwf**
    - **Objectif** : Exploration initiale du dataset C-MAPSS.
    - **Entrées** :
      - Fichiers `train_FD00X.txt` depuis `datasets/nasa-cmapss/`.
    - **Étapes** :
      - Lecture des fichiers texte.
      - Nettoyage/typage des colonnes (id moteur, cycle, capteurs, operating conditions).
      - Statistiques descriptives (min, max, moyenne, std).
      - Visualisations de base (histogrammes, boxplots, courbes temps/cycle).
    - **Sorties** :
      - Table KNIME nettoyée.
      - Export éventuel en `.csv` dans `data-mining/outputs/exploration/`.

 2. **02_statistical_feature_analysis.knwf**
    - **Objectif** : Analyse statistique des features (brutes et/ou extraites).
    - **Entrées** :
      - Données issues de `01_exploration_cmapss.knwf`.
      - Optionnel : features extraites depuis TimescaleDB ou fichiers exportés par le service `extraction-features`.
    - **Étapes** :
      - Calcul de statistiques par capteur / feature.
      - Analyse de corrélations (matrice de corrélation, heatmaps).
      - Analyse de distributions et détection de valeurs extrêmes.
    - **Sorties** :
      - Rapports de corrélations.
      - Tables agrégées dans `data-mining/outputs/statistics/`.

 3. **03_feature_selection.knwf**
    - **Objectif** : Sélection de features pertinentes pour les modèles de **détection d’anomalies** et de **RUL**.
    - **Entrées** :
      - Features agrégées/statistiques issues de `02_statistical_feature_analysis.knwf`.
      - Labels ou RUL (pour les scénarios où la RUL est disponible).
    - **Étapes** :
      - Filtrage par corrélation (features trop corrélées entre elles).
      - Tests de pertinence par rapport à la RUL (corrélation, mutual information).
      - Optionnel : méthodes de sélection wrapper/embedded (Random Forest, etc. via KNIME).
    - **Sorties** :
      - Liste des features sélectionnées.
      - Table réduite sauvegardée dans `data-mining/outputs/feature-selection/`.

 4. **04_ml_data_preparation.knwf**
    - **Objectif** : Préparer les données finales pour l’entraînement des modèles ML/DL (phases 6 et 7).
    - **Entrées** :
      - Table de features filtrées issue de `03_feature_selection.knwf`.
      - RUL (fichiers `RUL_FD00X.txt`) pour les scénarios de prédiction RUL.
    - **Étapes** :
      - Jointure des tables features + RUL.
      - Création de fenêtres temporelles si nécessaire (séquences pour LSTM/GRU).
      - Normalisation/standardisation (en cohérence avec la Phase 4).
      - Split train/validation/test.
    - **Sorties** :
      - Datasets prêts pour ML :
        - `data-mining/outputs/ml/train.csv`
        - `data-mining/outputs/ml/val.csv`
        - `data-mining/outputs/ml/test.csv`

 ---

 ## Organisation des dossiers

 À créer / utiliser autour des workflows KNIME :

 - `data-mining/knime-workflows/`
   - `PHASE_5_KNIME_PLAN.md` (ce fichier)
   - `README.md` (vue d’ensemble des workflows)
 - `data-mining/outputs/`
   - `exploration/`
   - `statistics/`
   - `feature-selection/`
   - `ml/`

 > Les sous-dossiers `outputs/*` peuvent être créés manuellement ou par les nœuds KNIME d’export de fichiers.

 ---

 ## Lien avec les autres phases

 - **Phase 3 – Prétraitement** :
   - Peut fournir des données prétraitées (fenêtres) à analyser dans KNIME.
 - **Phase 4 – ExtractionFeatures** :
   - Les features extraites (stockées dans TimescaleDB ou exportées en fichiers) peuvent être importées dans KNIME pour affiner la sélection de features.
 - **Phase 6 – DétectionAnomalies** & **Phase 7 – PrédictionRUL** :
   - Utiliseront les datasets préparés dans `data-mining/outputs/ml/` pour entraîner et évaluer les modèles.

 ---

 ## Checklist Phase 5 (KNIME)

 - [ ] Créer les workflows KNIME suivants :
   - [ ] `01_exploration_cmapss.knwf`
   - [ ] `02_statistical_feature_analysis.knwf`
   - [ ] `03_feature_selection.knwf`
   - [ ] `04_ml_data_preparation.knwf`
 - [ ] Documenter chaque workflow (description, entrées, sorties).
 - [ ] Configurer les chemins vers `datasets/nasa-cmapss/`.
 - [ ] Générer les exports dans `data-mining/outputs/`.
 - [ ] Vérifier la reproductibilité des résultats.

 ---

 ## Prochaines actions concrètes

 1. Ouvrir **KNIME Analytics Platform**.
 2. Créer un nouveau workspace KNIME pointant vers ce dépôt.
 3. Créer progressivement les 4 workflows `.knwf` en suivant ce plan.
 4. Tester les workflows avec `train_FD001.txt` dans un premier temps, puis étendre aux autres sous-ensembles (FD002–FD004).
 5. Versionner les fichiers `.knwf` dans `data-mining/knime-workflows/` (commit + push).


