# 02_statistical_feature_analysis.knwf – Guide détaillé pas à pas (KNIME débutant)

Ce fichier t’explique **clic par clic** comment construire le workflow `02_statistical_feature_analysis` dans KNIME pour analyser statistiquement les capteurs et leurs corrélations.

---

## 0. Préparation dans KNIME

1. **Ouvrir KNIME Analytics Platform**.
2. Dans le volet de gauche **KNIME Explorer** :
   - Clique droit sur ton espace de travail (ex. `LOCAL (Local Workspace)`).
   - Choisis **New → New KNIME Workflow…**.
   - Donne le nom : `02_statistical_feature_analysis`.
   - Clique sur **Finish**.
3. Tu arrives sur un **canvas vide** : c’est là que tu vas déposer les nodes.

**Note :** Ce workflow peut utiliser soit :
- Les données exportées du workflow 01 (fichier CSV dans `data-mining/outputs/exploration/exploration_train_FD001_raw.csv`).
- Ou directement le fichier `train_FD001.txt` (comme dans le workflow 01).

Pour ce guide, on va utiliser directement `train_FD001.txt` pour être autonome.

---

## 1. Lecture des données – Node `File Reader`

**Objectif :** lire le fichier `train_FD001.txt` (ou le CSV exporté du workflow 01).

### Option A : Lire depuis `train_FD001.txt` (comme workflow 01)

1. Dans le **Node Repository**, cherche **"File Reader"**.
2. Glisse-dépose `File Reader` sur le canvas.
3. **Configurer** :
   - Clique droit → **Configure…**.
   - **Browse…** → navigue vers `datasets/nasa-cmapss/train_FD001.txt`.
   - **Next**.
4. Options :
   - **Column header** : **No column header** (décocher).
   - **Column delimiter** : **Space** (ou `Space(s)`).
5. **Adjust column types** : toutes en **Double**.
6. **Finish**.
7. Exécute : clique droit → **Execute**.

### Option B : Lire depuis le CSV exporté du workflow 01

1. Dans le **Node Repository**, cherche **"CSV Reader"** (ou `File Reader` avec format CSV).
2. Glisse-dépose sur le canvas.
3. **Configurer** :
   - **Browse…** → `data-mining/outputs/exploration/exploration_train_FD001_raw.csv`.
   - **Column header** : **Has column header** (cocher, car le CSV a des en-têtes).
   - **Column delimiter** : **Comma**.
4. **Finish** et **Execute**.

**Pour la suite, on suppose que tu as les colonnes : `id`, `cycle`, `op1`, `op2`, `op3`, `s1`, `s2`, …, `s21`.**

---

## 2. Renommage des colonnes (si nécessaire) – Node `Column Rename`

**Si tu utilises Option A** (lecture directe de `train_FD001.txt`), tu dois renommer les colonnes comme dans le workflow 01.

1. **Node Repository** → cherche **"Column Rename"** (ou `Manipulation → Column → Rename`).
2. Glisse-dépose sur le canvas.
3. **Connecter** : tire une flèche depuis `File Reader` vers `Column Rename`.
4. **Configurer** :
   - Clique droit sur `Column Rename` → **Configure…**.
   - Dans la liste des colonnes, pour chaque colonne :
     - **Col 0** → renomme en `id`.
     - **Col 1** → renomme en `cycle`.
     - **Col 2** → renomme en `op1`.
     - **Col 3** → renomme en `op2`.
     - **Col 4** → renomme en `op3`.
     - **Col 5** → renomme en `s1`.
     - **Col 6** → renomme en `s2`.
     - … (continue jusqu’à **Col 25** → `s21`).
   - Pour renommer : double-clique sur le nom dans la colonne **"New Name"**, ou sélectionne et tape le nouveau nom.
5. **OK**.
6. Exécute : clique droit → **Execute**.

**Si tu utilises Option B** (CSV avec en-têtes), tu peux sauter cette étape.

---

## 3. Filtrer les colonnes capteurs – Node `Column Filter`

**Objectif :** créer une vue avec **uniquement les colonnes capteurs** (`s1` à `s21`) pour l’analyse statistique.

1. **Node Repository** → cherche **"Column Filter"** (`Manipulation → Column → Filter`).
2. Glisse-dépose sur le canvas.
3. **Connecter** : tire une flèche depuis `Column Rename` (ou `File Reader` si tu as sauté l’étape 2) vers `Column Filter`.
4. **Configurer** :
   - Clique droit → **Configure…**.
   - Dans la liste **"Available columns"**, tu vois toutes les colonnes.
   - **Sélectionne uniquement les colonnes capteurs** :
     - Clique sur `s1`, puis **Shift + clic** sur `s21` pour sélectionner toutes les colonnes de `s1` à `s21`.
     - Clique sur le bouton **`>>`** (ou **"Include"**) pour les déplacer vers **"Included columns"**.
   - Vérifie que **seulement** `s1`, `s2`, …, `s21` sont dans **"Included columns"**.
   - Les autres colonnes (`id`, `cycle`, `op1`, `op2`, `op3`) doivent rester dans **"Available columns"**.
5. **OK**.
6. Exécute : clique droit → **Execute**.

**Résultat :** tu as maintenant une table avec uniquement les 21 colonnes capteurs.

---

## 4. Vérifier les données avant corrélation (optionnel mais recommandé) – Node `Missing Value`

**Objectif :** vérifier et gérer les valeurs manquantes avant de calculer les corrélations.

1. **Node Repository** → cherche **"Missing Value"** (`Manipulation → Column → Missing Value`).
2. Glisse-dépose sur le canvas.
3. **Connecter** : tire une flèche depuis `Column Filter` vers `Missing Value`.
4. **Configurer** :
   - Clique droit → **Configure…**.
   - **Column Selection** : sélectionne toutes les colonnes capteurs (`s1` à `s21`).
   - **Missing value handling** :
     - Choisis **"Skip"** (ignorer les lignes avec valeurs manquantes) ou **"Mean imputation"** (remplacer par la moyenne).
   - **OK**.
5. Exécute : clique droit → **Execute**.

**Note :** Si tu n’as pas de valeurs manquantes dans tes données (ce qui est normal pour NASA C-MAPSS), tu peux sauter cette étape. Mais si tu vois des avertissements, cette étape peut aider.

---

## 5. Calculer la matrice de corrélation – Node `Linear Correlation`

**Objectif :** calculer les corrélations entre tous les capteurs.

1. **Node Repository** → cherche **"Linear Correlation"** (`Analytics → Statistics → Linear Correlation`).
2. Glisse-dépose sur le canvas.
3. **Connecter** : tire une flèche depuis `Missing Value` (ou `Column Filter` si tu as sauté l’étape 4) vers `Linear Correlation`.
4. **Configurer** :
   - Clique droit → **Configure…**.
   - **Column Selection** :
     - Dans **"Available columns"**, sélectionne toutes les colonnes capteurs (`s1` à `s21`).
     - Clique sur **`>>`** pour les déplacer vers **"Included columns"**.
   - **Output column pairs** :
     - Choisis **"Include only column pairs of compatible columns"** (c’est ce que tu as fait, c’est correct).
   - **Possible values count** :
     - Mettre **50** est OK (ou laisse la valeur par défaut).
   - **P-value** :
     - **"Two-sided"** est correct (c’est ce que tu as choisi).
   - **Correlation method** :
     - Par défaut, KNIME utilise **Pearson** (même si ce n’est pas visible dans l’interface, c’est bien Pearson).
   - **Missing value handling** :
     - Si disponible, choisis **"Skip"** (ignorer les valeurs manquantes).
5. **OK**.
6. Exécute : clique droit → **Execute**.

**Si tu vois un avertissement (icône jaune) :**
- Clique sur le node `Linear Correlation` pour voir le message d’avertissement.
- Les valeurs manquantes peuvent apparaître si :
  - Certaines colonnes ont des valeurs constantes (écart-type = 0) → la corrélation ne peut pas être calculée.
  - Certaines colonnes ont trop de valeurs identiques.
- **Solution :** Vérifie les données en amont avec un node `Statistics` pour voir quelles colonnes ont un écart-type = 0.

**Résultat :** le node produit une **table de corrélation** avec les corrélations entre chaque paire de capteurs.

**Visualiser le résultat :** clique droit sur `Linear Correlation` → **"Linear Correlation" → "Correlation matrix"** pour voir la table.

**Note importante :** Les valeurs manquantes dans la matrice de corrélation sont **normales** si certaines colonnes sont constantes ou ont une variance nulle. Tu peux continuer avec la heatmap même s’il y a quelques valeurs manquantes.

---

## 6. Préparer la matrice de corrélation pour visualisation – Node `Pivoting`

**Objectif :** transformer la sortie de `Linear Correlation` en format adapté pour la heatmap.

**Note :** Le node `Linear Correlation` produit une table avec des colonnes comme `Column 1`, `Column 2`, `Correlation`, etc. Pour la heatmap, on a besoin d’une matrice carrée (lignes = capteurs, colonnes = capteurs, valeurs = corrélations).

1. **Node Repository** → cherche **"Pivoting"** (`Manipulation → Row → Pivoting`).
2. Glisse-dépose sur le canvas.
3. **Connecter** : tire une flèche depuis `Linear Correlation` vers `Pivoting`.
4. **Configurer** :
   - Clique droit → **Configure…**.
   - **Group column(s)** :
     - Sélectionne la colonne qui contient le nom du premier capteur (probablement **"Column 1"** ou **"First Column"**).
   - **Pivot column** :
     - Sélectionne la colonne qui contient le nom du deuxième capteur (probablement **"Column 2"** ou **"Second Column"**).
   - **Aggregation** :
     - Choisis la colonne **"Correlation"** (ou **"Correlation Coefficient"**).
     - Méthode d’agrégation : **"First"** ou **"Mean"** (il n’y a qu’une seule valeur par paire).
   - **OK**.
5. Exécute : clique droit → **Execute**.

**Alternative plus simple :** Si le format de sortie de `Linear Correlation` est déjà une matrice (lignes = capteurs, colonnes = capteurs), tu peux **sauter cette étape** et passer directement à l’étape 7.

---

## 7. Visualiser la matrice de corrélation – Node `Heatmap (JavaScript Legacy)` ou `Interactive Table`

**Objectif :** créer une heatmap colorée pour visualiser les corrélations.

### Option A : Utiliser `Heatmap (JavaScript Legacy)` (ce que tu as choisi)

1. **Node Repository** → cherche **"Heatmap (JavaScript Legacy)"** (`Views → JavaScript Views → Heatmap (JavaScript Legacy)`).
2. Glisse-dépose sur le canvas.
3. **Connecter** : tire une flèche depuis `Pivoting` (ou `Linear Correlation` si tu as sauté l’étape 6) vers `Heatmap (JavaScript Legacy)`.
4. **Configurer** :
   - Clique droit → **Configure…**.
   - **Column Selection** :
     - Dans **"Available columns"**, sélectionne **toutes les colonnes de capteurs** (`s1` à `s21`).
     - Clique sur **`>>`** pour les déplacer vers **"Included columns"**.
     - **Important :** Assure-toi que la première colonne (celle qui contient les noms de capteurs en lignes) est **exclue** de la sélection. Tu veux seulement les colonnes numériques de corrélation.
   - **Color scheme** :
     - Choisis un schéma de couleurs (ex. **"Red-Yellow-Green"** ou **"Blue-White-Red"**).
   - **Value range** :
     - **Min** : `-1` (corrélation minimale).
     - **Max** : `1` (corrélation maximale).
   - **OK**.
5. Exécute : clique droit → **Execute**.

**Si tu vois "image port inactive port object" :**
- Cela signifie que le node attend des données sur un **port d’image**, mais `Linear Correlation` ne produit pas d’image, seulement une table.
- **Solution :** Utilise plutôt l’**Option B** ci-dessous.

### Option B : Utiliser `Interactive Table` avec formatage conditionnel (RECOMMANDÉ)

1. **Node Repository** → cherche **"Interactive Table"** (`Views → JavaScript Views → Interactive Table`).
2. Glisse-dépose sur le canvas.
3. **Connecter** : tire une flèche depuis `Linear Correlation` (ou `Pivoting`) vers `Interactive Table`.
4. **Configurer** :
   - Clique droit → **Configure…**.
   - **Column Selection** : sélectionne toutes les colonnes de corrélation.
   - **OK**.
5. Exécute : clique droit → **Execute**.

**Visualiser :** clique droit sur `Interactive Table` → **"Interactive Table" → "View: Interactive Table"**.

Dans la vue interactive, tu peux :
- Appliquer un **formatage conditionnel** (couleurs) sur les valeurs de corrélation.
- Clique sur une colonne → **Format → Conditional Formatting** → choisis un schéma de couleurs.

### Option C : Utiliser `Color Manager` + `Table View` (alternative simple)

1. **Node Repository** → cherche **"Color Manager"** (`Views → Color Manager`).
2. Glisse-dépose sur le canvas.
3. **Connecter** : tire une flèche depuis `Linear Correlation` vers `Color Manager`.
4. **Configurer** :
   - Clique droit → **Configure…**.
   - **Column Selection** : sélectionne toutes les colonnes de corrélation.
   - **Color scheme** : choisis **"Red-Yellow-Green"** ou **"Blue-White-Red"**.
   - **Value range** : Min = `-1`, Max = `1`.
   - **OK**.
5. Exécute : clique droit → **Execute**.
6. Ajoute un node **"Table View"** après `Color Manager` pour voir la table colorée.

**Visualiser :** clique droit sur `Table View` → **"Table View" → "View: Table"** pour voir la table avec les couleurs.

**Interprétation :**
- Les zones **rouges/bleues foncées** indiquent des capteurs très corrélés (redondants).
- Les zones **blanches/jaunes** indiquent des capteurs peu corrélés (indépendants).

**Recommandation :** Utilise l’**Option C** (`Color Manager` + `Table View`) car c’est le plus simple et le plus fiable.

---

## 8. Calculer des statistiques avancées – Node `Statistics`

**Objectif :** calculer des statistiques détaillées pour chaque capteur (moyenne, écart-type, variance, skewness, kurtosis, etc.).

1. **Node Repository** → cherche **"Statistics"** (`Analytics → Statistics → Statistics`).
2. Glisse-dépose sur le canvas.
3. **Connecter** : tire une flèche depuis `Column Filter` (sortie avec colonnes capteurs) vers `Statistics`.
   - **Note :** on utilise la sortie de `Column Filter` (pas `Linear Correlation`), car on veut les données brutes, pas la matrice de corrélation.
4. **Configurer** :
   - Clique droit → **Configure…**.
   - **Column Selection** :
     - Sélectionne toutes les colonnes capteurs (`s1` à `s21`).
   - **Statistics to compute** :
     - Coche les statistiques que tu veux :
       - ✅ **Mean** (moyenne)
       - ✅ **Standard deviation** (écart-type)
       - ✅ **Variance**
       - ✅ **Min** (minimum)
       - ✅ **Max** (maximum)
       - ✅ **Median** (médiane)
       - ✅ **25th percentile** (quartile Q1)
       - ✅ **75th percentile** (quartile Q3)
       - ✅ **Skewness** (asymétrie)
       - ✅ **Kurtosis** (aplatissement)
       - ✅ **Count** (nombre de valeurs)
       - ✅ **Missing values** (valeurs manquantes)
5. **OK**.
6. Exécute : clique droit → **Execute**.

**Visualiser les statistiques :** clique droit sur `Statistics` → **"Statistics" → "Statistics table"** pour voir le tableau récapitulatif.

**Interprétation :**
- **Skewness** proche de 0 = distribution symétrique.
- **Skewness > 0** = distribution décalée vers la droite (queue à droite).
- **Kurtosis** proche de 3 = distribution normale.
- **Kurtosis > 3** = distribution plus pointue que la normale.

---

## 9. Analyse PCA (optionnel) – Node `PCA`

**Objectif :** analyser la variance expliquée et identifier les composantes principales pour réduire la dimensionnalité.

1. **Node Repository** → cherche **"PCA"** (`Analytics → Dimension Reduction → PCA`).
2. Glisse-dépose sur le canvas.
3. **Connecter** : tire une flèche depuis `Column Filter` (sortie avec colonnes capteurs) vers `PCA`.
4. **Configurer** :
   - Clique droit → **Configure…**.
   - **Column Selection** :
     - Sélectionne toutes les colonnes capteurs (`s1` à `s21`).
   - **Number of components** :
     - Laisse **"Automatic"** (KNIME choisira le nombre optimal).
     - Ou fixe un nombre (ex. **5** ou **10**) si tu veux limiter.
   - **Normalize** :
     - ✅ **Coche "Normalize"** (recommandé pour que toutes les variables aient la même échelle).
   - **Missing value handling** :
     - Choisis **"Skip"** ou **"Mean imputation"**.
5. **OK**.
6. Exécute : clique droit → **Execute**.

**Visualiser les résultats :**
- **Variance expliquée :** clique droit → **"PCA" → "Eigenvalues"** pour voir combien de variance est expliquée par chaque composante.
- **Composantes principales :** clique droit → **"PCA" → "Transformed data"** pour voir les données projetées sur les nouvelles composantes.

**Interprétation :**
- Si les **2-3 premières composantes** expliquent **> 80% de la variance**, tu peux réduire la dimensionnalité sans perdre trop d’information.
- Cela peut aider à identifier des groupes de capteurs redondants.

---

## 10. Exporter les statistiques – Node `CSV Writer`

**Objectif :** sauvegarder la table de statistiques dans un fichier CSV.

1. **Node Repository** → cherche **"CSV Writer"** (`IO → Write → CSV Writer`).
2. Glisse-dépose sur le canvas.
3. **Connecter** : tire une flèche depuis `Statistics` (sortie de statistiques) vers `CSV Writer`.
4. **Configurer** :
   - Clique droit → **Configure…**.
   - **Output file** :
     - Clique sur **Browse…**.
     - Navigue vers le dossier `data-mining/outputs/statistics/`.
     - Si le dossier n’existe pas, crée-le manuellement dans ton explorateur de fichiers.
     - Nomme le fichier : `statistics_train_FD001_sensors.csv`.
   - **Column delimiter** :
     - Choisis **Comma**.
   - **Write column header** :
     - ✅ **Coche** (pour avoir les noms de colonnes).
   - **Quote character** :
     - Laisse **Double quote** (par défaut).
5. **OK**.
6. Exécute : clique droit → **Execute**.

**Vérification :** ouvre le fichier `data-mining/outputs/statistics/statistics_train_FD001_sensors.csv` dans Excel ou un éditeur de texte pour vérifier.

---

## 11. Exporter la matrice de corrélation – Node `CSV Writer`

**Objectif :** sauvegarder la matrice de corrélation dans un fichier CSV.

1. **Node Repository** → cherche **"CSV Writer"**.
2. Glisse-dépose sur le canvas (un deuxième `CSV Writer`).
3. **Connecter** : tire une flèche depuis `Linear Correlation` (sortie de corrélation) vers ce nouveau `CSV Writer`.
4. **Configurer** :
   - Clique droit → **Configure…**.
   - **Output file** :
     - **Browse…** → `data-mining/outputs/statistics/correlation_train_FD001_sensors.csv`.
   - **Column delimiter** : **Comma**.
   - **Write column header** : ✅ **Coche**.
5. **OK**.
6. Exécute : clique droit → **Execute**.

**Vérification :** ouvre le fichier `data-mining/outputs/statistics/correlation_train_FD001_sensors.csv` pour voir la matrice de corrélation.

---

## 12. Structure finale du workflow

Ton workflow devrait ressembler à ceci (chaîne de nodes) :

```
File Reader
    ↓
Column Rename (si nécessaire)
    ↓
Column Filter (capteurs seulement)
    ├─→ Missing Value (optionnel)
    │       ↓
    │   Linear Correlation → Pivoting (optionnel) → Color Manager → Table View
    │                           ↓
    │                       CSV Writer (corrélation)
    │
    ├─→ Statistics → CSV Writer (statistiques)
    │
    └─→ PCA (optionnel)
```

**Connexions :**
- `File Reader` → `Column Rename` (si nécessaire)
- `Column Rename` → `Column Filter`
- `Column Filter` → `Missing Value` (optionnel) → `Linear Correlation` → `Pivoting` (optionnel) → `Color Manager` → `Table View`
- `Linear Correlation` → `CSV Writer` (corrélation)
- `Column Filter` → `Statistics` → `CSV Writer` (statistiques)
- `Column Filter` → `PCA` (optionnel)

---

## 13. Exécuter tout le workflow

1. **Exécuter tous les nodes** :
   - Clique droit sur le canvas (zone vide) → **Execute All**.
   - Ou exécute chaque node un par un (clique droit → **Execute**).
2. **Vérifier les erreurs** :
   - Si un node est **rouge**, il y a une erreur. Clique dessus pour voir le message d’erreur.
   - Si un node est **jaune**, il attend une exécution.
   - Si un node est **vert**, il a réussi.
3. **Visualiser les résultats** :
   - Clique droit sur chaque node → **"Node name" → "Data table"** (ou **"View"**) pour voir les résultats.

---

## 14. Sauvegarder le workflow (`.knwf`)

1. Dans KNIME, va dans **File → Save** (ou **Ctrl+S**).
2. KNIME sauvegarde automatiquement le workflow dans ton workspace.
3. Le fichier `.knwf` sera créé dans :
   - `data-mining/knime-workflows/02_statistical_feature_analysis.knwf`
   - (si ton workspace KNIME pointe vers le dossier du projet Git).

**Pour retrouver le fichier :**
- Dans **KNIME Explorer**, tu devrais voir `02_statistical_feature_analysis` dans ton workspace.
- Clique droit → **"Open in File System"** pour voir l’emplacement exact.

---

## 15. Résultats attendus

Après exécution, tu devrais avoir :

1. **Table de statistiques** (`statistics_train_FD001_sensors.csv`) :
   - Une ligne par capteur (`s1` à `s21`).
   - Colonnes : Mean, Std Dev, Variance, Min, Max, Median, Q1, Q3, Skewness, Kurtosis, etc.

2. **Matrice de corrélation** (`correlation_train_FD001_sensors.csv`) :
   - Une matrice 21×21 avec les corrélations entre chaque paire de capteurs.
   - Valeurs entre -1 et 1.

3. **Heatmap visuelle** (dans KNIME) :
   - Visualisation colorée des corrélations.

4. **Résultats PCA** (si tu as ajouté le node) :
   - Variance expliquée par composante.
   - Données transformées.

---

## 16. Prochaines étapes

Une fois ce workflow terminé, tu peux :
- Passer au workflow **03_feature_selection** pour sélectionner les features les plus pertinentes.
- Utiliser les résultats de corrélation pour identifier les capteurs redondants.
- Utiliser les statistiques pour détecter des anomalies ou des distributions anormales.

---

## Aide et dépannage

**Problème : "Column not found"**
- Vérifie que tu as bien renommé les colonnes dans l’étape 2.
- Vérifie que les noms de colonnes correspondent exactement (`s1`, `s2`, etc.).

**Problème : "Missing values" dans Linear Correlation**
- Les valeurs manquantes dans la matrice de corrélation sont **normales** si certaines colonnes ont une variance nulle (valeurs constantes).
- Vérifie avec un node `Statistics` quelles colonnes ont un écart-type = 0.
- Tu peux ignorer ces valeurs manquantes et continuer.

**Problème : "Avertissement (icône jaune)" sur Linear Correlation**
- Clique sur le node pour voir le message d’avertissement.
- Si c’est à cause de colonnes constantes, c’est normal. Tu peux continuer.

**Problème : "Heatmap JavaScript Legacy - image port inactive port object"**
- Ce node attend une image, pas une table. Utilise plutôt **`Color Manager` + `Table View`** (Option C de l’étape 7).
- Ou utilise **`Interactive Table`** avec formatage conditionnel (Option B).

**Problème : "Heatmap node not found"**
- Cherche **"Color Manager"** (`Views → Color Manager`) comme alternative.
- Ou utilise **"Interactive Table"** (`Views → JavaScript Views → Interactive Table`).

**Problème : "PCA node not found"**
- Le node PCA peut nécessiter une extension KNIME.
- Va dans **File → Preferences → KNIME → KNIME Extensions** et installe **"KNIME Analytics Platform"** ou **"Dimension Reduction"** si nécessaire.

---

**Félicitations !** Tu as terminé le workflow 02. Tu peux maintenant passer au workflow 03 pour la sélection de features.
