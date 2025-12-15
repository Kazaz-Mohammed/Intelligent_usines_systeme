# Stratégie Git et GitHub - Workflow de Développement

## Vue d'Ensemble

Ce document définit la stratégie de versioning et de workflow Git pour le projet. Le respect de cette stratégie garantit :
- ✅ Historique clair et traçable
- ✅ Possibilité de rollback en cas de problème
- ✅ Collaboration efficace (si travail en équipe)
- ✅ Déploiement sécurisé

---

## Structure des Branches

### Branches Principales

```
main
├── Production-ready code uniquement
├── Protégée (merge uniquement via PR)
└── Tags de releases (v1.0.0, v1.1.0, etc.)

develop
├── Branche de développement principale
├── Intégration des features
└── Tests et validation avant merge dans main
```

### Branches de Développement

```
feature/[nom-service] ou feature/[nom-fonctionnalité]
├── Exemples :
│   ├── feature/infrastructure-docker
│   ├── feature/service-ingestion-iiot
│   ├── feature/service-prediction-rul
│   └── feature/data-mining-knime
├── Créée depuis develop
└── Mergée dans develop après validation

bugfix/[description]
├── Corrections de bugs
├── Créée depuis develop ou main
└── Mergée rapidement après correction

hotfix/[description]
├── Corrections urgentes sur main
├── Créée depuis main
├── Mergée dans main ET develop
└── Pour problèmes critiques en production
```

---

## Workflow de Développement

### 1. Démarrage d'une Nouvelle Phase/Fonctionnalité

```bash
# 1. S'assurer d'être à jour
git checkout develop
git pull origin develop

# 2. Créer nouvelle branche feature
git checkout -b feature/nom-du-service

# 3. Développer et commit fréquent
git add .
git commit -m "[feat][nom-service] Description de la fonctionnalité"

# 4. Push régulier (minimum quotidien)
git push origin feature/nom-du-service
```

### 2. Pendant le Développement

**Règle d'or** : **Push minimum 1 fois par jour**, même si le code n'est pas complet.

```bash
# Après chaque fonctionnalité ou correction testée
git add .
git commit -m "[type][service] Description claire"
git push origin feature/nom-du-service
```

**Avantages** :
- Sauvegarde automatique sur GitHub
- Possibilité de continuer sur autre machine
- Historique détaillé des changements

### 3. Fin de Développement d'une Phase

```bash
# 1. Vérifier que tous les tests passent
# 2. Vérifier la documentation
# 3. Commit final si nécessaire
git add .
git commit -m "[feat][service] Phase complète - Ready for review"

# 4. Push final
git push origin feature/nom-du-service

# 5. Créer Pull Request sur GitHub
# (via interface GitHub ou CLI)
```

### 4. Merge dans develop

**Via Pull Request (recommandé)** :
1. Aller sur GitHub
2. Créer Pull Request : `feature/nom-service` → `develop`
3. Reviewer le code (si travail en équipe)
4. Valider les tests CI/CD
5. Merge après validation

**Via ligne de commande** (si seul développeur) :
```bash
git checkout develop
git pull origin develop
git merge feature/nom-service
git push origin develop

# Supprimer branche feature (optionnel)
git branch -d feature/nom-service
git push origin --delete feature/nom-service
```

### 5. Tagging des Versions

**Après chaque phase complète** :

```bash
# 1. S'assurer d'être sur develop (ou main pour release)
git checkout develop

# 2. Créer tag
git tag -a v0.1.0 -m "Phase 1: Infrastructure Docker"

# 3. Push le tag
git push origin v0.1.0
```

**Convention de versioning** :
- `v0.1.0` : Phase 1 (Infrastructure)
- `v0.2.0` : Phase 2 (IngestionIIoT)
- `v0.3.0` : Phase 3 (Prétraitement)
- ...
- `v1.0.0` : Version finale complète

---

## Format des Messages de Commit

### Structure

```
[TYPE][SERVICE] Description courte (max 50 caractères)

Description détaillée si nécessaire (optionnel)
- Point 1
- Point 2

Fixes #issue_number (si applicable)
```

### Types de Commit

| Type | Usage | Exemple |
|------|-------|---------|
| `feat` | Nouvelle fonctionnalité | `[feat][ingestion-iiot] Ajout support OPC UA` |
| `fix` | Correction de bug | `[fix][preprocessing] Correction rééchantillonnage` |
| `docs` | Documentation uniquement | `[docs][prediction-rul] Ajout docstring modèles` |
| `test` | Ajout/modification de tests | `[test][detection-anomalies] Tests unitaires IsolationForest` |
| `refactor` | Refactoring (pas de changement fonctionnel) | `[refactor][dashboard] Réorganisation composants React` |
| `chore` | Maintenance (dépendances, config) | `[chore] Mise à jour dépendances Python` |
| `perf` | Amélioration performance | `[perf][extraction-features] Optimisation calcul RMS` |

### Services (tags)

| Tag | Service |
|-----|---------|
| `infra` | Infrastructure (Docker, K8s) |
| `ingestion-iiot` | Service IngestionIIoT |
| `preprocessing` | Service Prétraitement |
| `extraction-features` | Service ExtractionFeatures |
| `detection-anomalies` | Service DétectionAnomalies |
| `prediction-rul` | Service PrédictionRUL |
| `orchestrateur` | Service OrchestrateurMaintenance |
| `dashboard` | Service DashboardUsine |
| `data-mining` | Workflows KNIME |
| `integration` | Intégration E2E |

### Exemples de Commits

```bash
# Bon commit
git commit -m "[feat][ingestion-iiot] Ajout connecteur OPC UA avec Eclipse Milo"

# Commit avec description
git commit -m "[feat][prediction-rul] Implémentation modèle LSTM custom

- Architecture LSTM 2 couches (128, 64)
- Entraînement sur NASA C-MAPSS
- Métriques MAE < 10 cycles validées"

# Correction bug
git commit -m "[fix][preprocessing] Correction gestion valeurs NaN dans rééchantillonnage

Fixes issue avec données manquantes lors synchronisation multi-capteurs"

# Tests
git commit -m "[test][detection-anomalies] Ajout tests unitaires IsolationForest

Couverture > 80% validée"
```

---

## Stratégie de Push

### Push Fréquent (Recommandé)

**Règle** : Push au minimum **1 fois par jour**, même si code incomplet.

**Avantages** :
- Sauvegarde automatique sur GitHub
- Possibilité de rollback facile
- Continuation sur autre machine
- Historique détaillé

### Quand Push ?

✅ **Toujours push dans ces cas** :
- Fin de session de travail
- Fonctionnalité testée et validée
- Avant changement majeur
- Avant expérimentation risquée

❌ **Ne pas push** :
- Code avec erreurs critiques non résolues
- Code non testé qui casse les builds
- Secrets/credentials (utiliser .gitignore)

### Workflow Push Quotidien

```bash
# Fin de journée
git status                    # Vérifier changements
git add .                     # Ajouter tous les changements
git commit -m "[feat][...] ..."  # Commit descriptif
git push origin feature/nom-service  # Push sur GitHub
```

---

## Stratégie de Rollback

### Si Problème sur Branche Feature

```bash
# Option 1 : Revert dernier commit
git revert HEAD
git push origin feature/nom-service

# Option 2 : Reset au commit précédent (DANGEREUX si déjà push)
git reset --hard HEAD~1
git push --force origin feature/nom-service  # ATTENTION : force push

# Option 3 : Revenir à un tag spécifique
git checkout v0.2.0
git checkout -b feature/recovery-from-v0.2.0
```

### Si Problème sur develop/main

```bash
# Identifier le commit problématique
git log --oneline

# Revert un commit spécifique
git revert <commit-hash>
git push origin develop

# OU revenir à un tag précédent
git checkout v0.5.0
git checkout -b hotfix/rollback-from-v0.6.0
# Corriger et merge
```

### Si Problème Docker/Kubernetes

```bash
# Rebuild depuis tag précédent
git checkout v0.3.0
docker-compose build
docker-compose up

# OU pour Kubernetes
kubectl rollout undo deployment/nom-service
```

---

## Protection des Branches

### Configuration Recommandée sur GitHub

**Branche `main`** :
- ✅ Protection activée
- ✅ Require pull request reviews (si équipe)
- ✅ Require status checks to pass (CI/CD)
- ✅ Require branches to be up to date
- ✅ Restrict pushes (push via PR uniquement)

**Branche `develop`** :
- ✅ Protection modérée
- ✅ Require pull request reviews (optionnel)
- ✅ Permettre merges directs si seul développeur

### Configuration via GitHub Web UI

1. Aller sur **Settings** → **Branches**
2. Cliquer **Add rule** pour `main`
3. Activer les protections nécessaires
4. Répéter pour `develop` si besoin

---

## Tags et Releases

### Création de Tags

```bash
# Tag annoté (recommandé)
git tag -a v0.1.0 -m "Phase 1: Infrastructure Docker complète"

# Tag simple (non recommandé pour releases)
git tag v0.1.0

# Push tag
git push origin v0.1.0

# Push tous les tags
git push origin --tags
```

### Releases GitHub

**Créer une Release sur GitHub** :

1. Aller sur **Releases** → **Draft a new release**
2. Sélectionner tag (ex: `v0.1.0`)
3. Titre : `Release v0.1.0 - Phase 1: Infrastructure`
4. Description :
   ```
   ## Phase 1: Infrastructure Docker
   
   ### Ajouté
   - Docker Compose avec Kafka, PostgreSQL, TimescaleDB, InfluxDB, MinIO
   - Scripts d'initialisation
   - Health checks configurés
   
   ### Tests
   - ✅ Tous les conteneurs démarrent
   - ✅ Health checks passent
   - ✅ Connectivité validée
   
   ### Prochaines étapes
   - Phase 2: Service IngestionIIoT
   ```

### Convention de Versioning (SemVer)

Format : `vMAJOR.MINOR.PATCH`

- **MAJOR** : Changements incompatibles (ex: v1.0.0 → v2.0.0)
- **MINOR** : Nouvelles fonctionnalités compatibles (ex: v0.1.0 → v0.2.0)
- **PATCH** : Corrections de bugs (ex: v0.1.0 → v0.1.1)

**Pour ce projet** :
- `v0.X.0` : Nouvelle phase complète
- `v0.X.Y` : Corrections dans une phase
- `v1.0.0` : Version finale complète

---

## Checklist Avant Push

Avant chaque push, vérifier :

- [ ] Code fonctionne localement
- [ ] Tests passent (au moins ceux existants)
- [ ] Pas de secrets/credentials dans le code
- [ ] `.gitignore` à jour
- [ ] Message de commit clair et descriptif
- [ ] Documentation mise à jour si nécessaire

---

## Checklist Avant Merge dans develop

Avant de merger une feature dans `develop` :

- [ ] Tous les tests passent (unitaires + intégration)
- [ ] Code review effectué (si équipe)
- [ ] Documentation à jour
- [ ] Pas de régressions
- [ ] Performance validée si applicable
- [ ] Commit final avec message approprié
- [ ] Tag créé si phase complète

---

## Exemple de Workflow Complet

### Scénario : Développement Service IngestionIIoT

```bash
# 1. Démarrage
git checkout develop
git pull origin develop
git checkout -b feature/service-ingestion-iiot

# 2. Développement (jour 1)
# ... code ...
git add .
git commit -m "[feat][ingestion-iiot] Structure Spring Boot initiale"
git push origin feature/service-ingestion-iiot

# 3. Développement (jour 2)
# ... code ...
git add .
git commit -m "[feat][ingestion-iiot] Ajout connecteur OPC UA"
git push origin feature/service-ingestion-iiot

# 4. Tests (jour 3)
# ... tests ...
git add .
git commit -m "[test][ingestion-iiot] Tests unitaires connecteur OPC UA"
git push origin feature/service-ingestion-iiot

# 5. Correction bug
# ... fix ...
git add .
git commit -m "[fix][ingestion-iiot] Correction gestion timeout OPC UA"
git push origin feature/service-ingestion-iiot

# 6. Finalisation
git add .
git commit -m "[feat][ingestion-iiot] Phase complète - Service fonctionnel"
git push origin feature/service-ingestion-iiot

# 7. Pull Request (via GitHub)
# ... créer PR sur GitHub ...

# 8. Merge (après validation)
git checkout develop
git pull origin develop
git merge feature/service-ingestion-iiot
git push origin develop

# 9. Tag
git tag -a v0.2.0 -m "Phase 2: Service IngestionIIoT complet"
git push origin v0.2.0

# 10. Nettoyage
git branch -d feature/service-ingestion-iiot
git push origin --delete feature/service-ingestion-iiot
```

---

## Commandes Git Essentielles

### Consultation

```bash
# Statut actuel
git status

# Historique commits
git log --oneline --graph --all

# Différences
git diff                    # Modifications non commitées
git diff HEAD~1            # Diff avec commit précédent
git diff v0.1.0 v0.2.0     # Diff entre tags

# Branches
git branch                  # Branches locales
git branch -a               # Toutes les branches
git branch -r               # Branches distantes
```

### Navigation

```bash
# Changer de branche
git checkout develop
git checkout -b feature/nouvelle-branche

# Aller à un tag/commit
git checkout v0.1.0
git checkout <commit-hash>
```

### Synchronisation

```bash
# Récupérer changements distants
git fetch origin

# Merge changements distants
git pull origin develop

# Push changements locaux
git push origin feature/nom-service
```

### Annulation

```bash
# Annuler modifications non commitées
git restore fichier.txt
git restore .

# Annuler staging
git restore --staged fichier.txt

# Revert commit (sans perte d'historique)
git revert HEAD
```

---

## Résumé : Règles d'Or

1. ✅ **Push minimum 1 fois par jour**
2. ✅ **Commit messages clairs et descriptifs**
3. ✅ **Tests avant merge dans develop**
4. ✅ **Tag après chaque phase complète**
5. ✅ **Ne jamais push secrets/credentials**
6. ✅ **Vérifier avant push** (tests, .gitignore)
7. ✅ **Documenter les décisions importantes**
8. ✅ **Utiliser branches feature** pour chaque développement
9. ✅ **Merge via Pull Request** quand possible
10. ✅ **Respecter la convention de commit**

---

## En Cas de Problème

**Erreur de push** :
- Vérifier connexion internet
- Vérifier permissions GitHub
- Essayer `git push --force-with-lease` (avec précaution)

**Conflit de merge** :
- Résoudre manuellement les conflits
- Tester après résolution
- Commit avec message `[fix] Résolution conflits merge`

**Perte de code** :
- Vérifier `git reflog` pour retrouver commits
- Récupérer depuis GitHub (historique en ligne)

**Questions** :
- Consulter documentation Git : https://git-scm.com/doc
- GitHub Help : https://docs.github.com

