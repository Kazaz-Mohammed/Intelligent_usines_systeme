# 01_exploration_cmapss.knwf – Guide détaillé pas à pas (KNIME débutant)

Ce fichier t’explique **clic par clic** comment construire le workflow `01_exploration_cmapss` dans KNIME, même si tu découvres la plateforme.

---

## 0. Préparation dans KNIME

1. **Ouvrir KNIME Analytics Platform**.
2. Dans le volet de gauche **KNIME Explorer** :
   - Clique droit sur ton espace de travail (ex. `LOCAL (Local Workspace)`).
   - Choisis **New → New KNIME Workflow…**.
   - Donne le nom : `01_exploration_cmapss`.
   - Clique sur **Finish**.
3. Tu arrives sur un **canvas vide** : c’est là que tu vas déposer les nodes.

Assure-toi que ton projet Git est accessible depuis KNIME, avec un chemin qui permet de pointer vers le dossier `datasets/nasa-cmapss/`.

---

## 1. Lecture des données – Node `File Reader`

**Objectif :** lire le fichier `train_FD001.txt` du dataset NASA C-MAPSS.

1. Dans la barre de menu, ouvre le **Node Repository** (souvent en bas à gauche).
2. Dans la barre de recherche du Node Repository, tape **"File Reader"**.
3. Prends le node `File Reader` (catégorie `IO → Read → File Reader`) et **glisse-dépose**-le sur le canvas.
4. **Configurer le node** :
   - Clique droit sur le node `File Reader` → **Configure…**.
   - Dans la fenêtre :
     - En haut, clique sur **Browse…** à côté du chemin.
     - Navigue jusqu’à ton projet, puis : `datasets/nasa-cmapss/train_FD001.txt`.
   - Clique sur **Next** (ou KNIME peut ouvrir directement une vue d’aperçu).
5. Dans les options de format :
   - **Column header** :
     - Décoche l’option indiquant que la première ligne contient les en-têtes (tu veux **No column header**).
   - **Column delimiter** :
     - Choisis **Space** (ou `Space(s)` selon la version).
   - KNIME va détecter un certain nombre de colonnes (26 au total pour FD001 : 1 id, 1 cycle, 3 conditions, 21 capteurs).
6. Clique sur **"Adjust column types"** si disponible :
   - Toutes les colonnes seront en général en **Double** (nombre).
   - C’est correct pour l’instant.
7. Clique sur **Finish**.

Tu peux maintenant **exécuter** le node : clique droit sur `File Reader` → **Execute**.  
Puis clique droit → **"File Reader" → "Data table"** pour voir les données brutes.

---

## 2. Renommage des colonnes – Node `Column Rename`

**Objectif :** donner des noms explicites aux colonnes (id, cycle, op1, …, s21).

1. Dans le Node Repository, cherche **"Column Rename"**.
2. Glisse le node `Column Rename` sur le canvas, à droite du `File Reader`.
3. Connecte la **sortie** du `File Reader` à l’**entrée** du `Column Rename` :
   - Clique sur le petit triangle à droite de `File Reader`, maintiens, et relâche sur le triangle gauche de `Column Rename`.
4. **Configurer le node** :
   - Clique droit sur `Column Rename` → **Configure…**.
   - Tu vois la liste des colonnes (`Column 0`, `Column 1`, etc. si tu n’as pas encore de noms).
   - Pour chaque colonne :
     - Sélectionne `Column 0` → dans la partie droite, **New name** = `id`.
     - Sélectionne `Column 1` → **New name** = `cycle`.
     - Sélectionne `Column 2` → **New name** = `op1`.
     - Sélectionne `Column 3` → **New name** = `op2`.
     - Sélectionne `Column 4` → **New name** = `op3`.
     - Sélectionne `Column 5` → **New name** = `s1`.
     - …
     - Sélectionne `Column 25` → **New name** = `s21`.
   - Valide avec **OK**.
5. Clique droit sur `Column Rename` → **Execute** pour appliquer.
6. Clique droit → **"Column Rename" → "Data table"** pour vérifier les nouveaux noms.

---

## 3. Filtrage / vues de colonnes – Nodes `Column Filter`

L’idée est de créer :
- une **vue complète** (toutes les colonnes),
- une **vue capteurs seulement** (s1…s21),
- éventuellement d’autres vues si tu veux (conditions seulement, etc.).

### 3.1 Vue complète – `Column Filter (all)`

1. Cherche `Column Filter` dans le Node Repository.
2. Glisse un node `Column Filter` sur le canvas, à droite de `Column Rename`.
3. Connecte la sortie de `Column Rename` à l’entrée du `Column Filter`.
4. Clique droit sur ce `Column Filter` → **Rename** (optionnel) et nomme-le : `Column Filter – All`.
5. **Configure…** :
   - Dans l’onglet principal :
     - Assure-toi que **toutes les colonnes** (`id`, `cycle`, `op1`, `op2`, `op3`, `s1`…`s21`) sont dans **Included columns**.
     - S’il y en a dans **Available**, utilise `>>` pour les ajouter à **Included**.
   - Clique sur **OK**.
6. Exécute le node (clic droit → **Execute**).

Ce node te sert simplement de **sortie de référence** avec toutes les colonnes.

### 3.2 Vue capteurs seulement – `Column Filter (sensors)`

1. Ajoute un **deuxième** node `Column Filter` sur le canvas.
2. Connecte **à nouveau** la sortie de `Column Rename` (ou de `Column Filter – All`) vers ce nouveau `Column Filter`.
3. Renomme-le en `Column Filter – Sensors` (clic droit → Rename).
4. **Configure…** :
   - Dans l’onglet principal :
     - Mets dans **Included columns** uniquement les colonnes `s1`, `s2`, …, `s21`.
       - Soit en les sélectionnant une par une depuis **Available** et en cliquant sur `>>`.
       - Soit en utilisant un **Name filter** si disponible (par ex. un motif `s*`) puis en ajoutant les colonnes filtrées.
     - Les colonnes `id`, `cycle`, `op1`, `op2`, `op3` doivent rester dans **Available**.
   - Clique sur **OK**.
5. Exécute le node.

Tu auras maintenant une sortie qui ne contient que les capteurs, pratique pour certaines visualisations.

---

## 4. Statistiques descriptives – Node `Statistics`

**Objectif :** voir min, max, moyenne, écart-type, etc. pour chaque colonne.

1. Dans le Node Repository, cherche **"Statistics"**.
2. Glisse le node `Statistics` sur le canvas, à droite de `Column Filter – All`.
3. Connecte la sortie de `Column Filter – All` à l’entrée du `Statistics`.
4. Clique droit sur `Statistics` → **Configure…** :
   - En général, tu peux laisser la configuration par défaut :
     - Toutes les colonnes numériques sont sélectionnées.
   - Vérifie simplement que l’onglet `Columns` (ou similaire) inclut bien les colonnes que tu veux analyser.
5. Exécute le node (clic droit → **Execute**).
6. Pour voir les résultats :
   - Clic droit sur `Statistics` → **Statistics** (ou **"Spec" / "Data table"** selon la version).

Tu verras un tableau avec pour chaque colonne : min, max, mean, quartiles, etc.

---

## 5. Histogramme – Node `Histogram`

**Objectif :** visualiser la distribution d’un capteur (par ex. `s2`).

1. Cherche `Histogram` dans le Node Repository.
2. Glisse `Histogram` sur le canvas, à droite de `Column Filter – Sensors` (c’est plus logique d’utiliser la vue capteurs).
3. Connecte la sortie de `Column Filter – Sensors` à l’entrée de `Histogram`.
4. Clique droit sur `Histogram` → **Configure…** :
   - Dans l’onglet `General` :
     - Choisis la colonne à analyser, par exemple **`s2`**.
   - Tu peux ajuster :
     - Le nombre de **bins** (barres) si nécessaire.
5. Exécute le node.
6. Clique droit → **View: Histogram** (ou similaire) pour voir la courbe.

---

## 6. Courbes temporelles – Nodes `Row Filter` + `Line Plot`

**Objectif :** tracer un capteur en fonction du cycle, éventuellement pour un moteur précis.

### 6.1 Filtrer un moteur – `Row Filter` (optionnel)

1. Cherche `Row Filter` dans le Node Repository.
2. Glisse `Row Filter` sur le canvas, entre `Column Filter – All` et `Line Plot` (que tu créeras juste après).
3. Connecte la sortie de `Column Filter – All` à l’entrée de `Row Filter`.
4. Clique droit sur `Row Filter` → **Configure…** :
   - Choisis l’option pour filtrer sur une **valeur égale**.
   - Colonne : `id`.
   - Valeur : `1` (par exemple, moteur 1).
   - Valide.
5. Exécute le node.

Tu as maintenant un sous-ensemble des données pour un seul moteur.

### 6.2 Tracer le signal – `Line Plot`

1. Cherche `Line Plot` dans le Node Repository.
2. Glisse `Line Plot` sur le canvas, à droite de `Row Filter` (ou directement de `Column Filter – All` si tu ne filtres pas par moteur).
3. Connecte la sortie de `Row Filter` (ou `Column Filter – All`) à l’entrée de `Line Plot`.
4. Clique droit sur `Line Plot` → **Configure…** :
   - Onglet `Data` :
     - **X-axis** : sélectionne la colonne `cycle`.
     - **Y-axis** : sélectionne un capteur, par ex. `s9`.
   - Tu peux choisir d’afficher une seule série ou plusieurs capteurs en même temps.
5. Exécute le node.
6. Clique droit → **View: Line Plot** pour visualiser la courbe.

---

## 7. Boîtes à moustaches – Node `Box Plot`

**Objectif :** visualiser la dispersion des capteurs (outliers, quartiles, etc.).

1. Cherche `Box Plot` dans le Node Repository.
2. Glisse `Box Plot` sur le canvas, à droite de `Column Filter – Sensors` (sortie capteurs).
3. Connecte `Column Filter – Sensors` → `Box Plot`.
4. Clique droit sur `Box Plot` → **Configure…** :
   - Dans l’onglet `Columns` :
     - Choisis un ou plusieurs capteurs (par ex. `s1`, `s2`, `s3`).
   - Tu peux également choisir un grouping si la version le permet (par ex. grouper par `id`).
5. Exécute le node.
6. Clique droit → **View: Box Plot** pour voir les boxplots.

---

## 8. Agrégation par moteur – Node `GroupBy` (optionnel)

**Objectif :** calculer des statistiques par moteur (moyenne, std des capteurs).

1. Cherche `GroupBy` dans le Node Repository.
2. Glisse `GroupBy` sur le canvas, à droite de `Column Filter – Sensors` (ou `Column Filter – All` si tu veux garder les conditions).
3. Connecte la sortie du filtre choisi au `GroupBy`.
4. Clique droit → **Configure…** :
   - Onglet **Groups** :
     - Ajoute la colonne `id` dans la liste des colonnes de grouping.
   - Onglet **Manual Aggregation** :
     - Ajoute les colonnes capteurs `s1`…`s21`.
     - Pour chaque colonne, choisis des fonctions d’agrégation :
       - `Mean` (moyenne),
       - `Standard deviation` (std),
       - éventuellement `Min`, `Max`.
5. Exécute le node.
6. Ouvre la table de sortie pour voir les statistiques par moteur.

---

## 9. Export des résultats – Node `CSV Writer` (optionnel)

Tu peux exporter deux types de tables :
- Le jeu de données brut typé (après `Column Filter – All` ou `Column Rename`).
- Le tableau agrégé par moteur (après `GroupBy`).

### 9.1 Export des données brutes

1. Cherche `CSV Writer` dans le Node Repository.
2. Glisse `CSV Writer` sur le canvas, à droite de `Column Filter – All`.
3. Connecte `Column Filter – All` → `CSV Writer`.
4. Clique droit → **Configure…** :
   - Choisis un chemin de sortie, par exemple :
     - `data-mining/outputs/exploration/exploration_train_FD001_raw.csv`
   - Vérifie le séparateur (`,` par défaut).
5. Exécute le node.

### 9.2 Export des statistiques par moteur

1. Ajoute un autre `CSV Writer` à droite du `GroupBy`.
2. Connecte `GroupBy` → `CSV Writer`.
3. Configure :
   - Chemin : `data-mining/outputs/exploration/exploration_train_FD001_agg_by_engine.csv`
4. Exécute le node.

---

## 10. Sauvegarder le workflow (`.knwf`)

1. Dans KNIME, menu **File → Save All** (ou l’icône de disquette).
2. KNIME crée automatiquement le dossier du workflow avec le fichier :
   - `01_exploration_cmapss.knwf`
3. Si ton workspace KNIME pointe sur le dossier du projet Git, tu retrouveras ce fichier dans :
   - `data-mining/knime-workflows/01_exploration_cmapss/` (structure interne de KNIME)
   - avec le `.knwf` à l’intérieur.

---

Si tu veux, tu peux m’envoyer un screenshot ou la **liste exacte des nodes** que tu as déjà posés, et je te dirai étape par étape quoi corriger ou compléter. 