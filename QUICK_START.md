# Guide de Démarrage Rapide

## 🎯 Résumé du Projet

Vous développez une **plateforme de maintenance prédictive** qui combine 3 modules :
- **ML/DL** : Modèles custom pour prédiction RUL et détection anomalies
- **Data Mining** : Analyse avec KNIME
- **Microservices** : Architecture Spring Boot + FastAPI

## 📋 Ce que vous devez faire MAINTENANT

### Étape 1 : Lire la Documentation

1. **Lire** `PROJECT_EXPLANATION.md` pour comprendre l'ensemble du projet
2. **Lire** `DEVELOPMENT_PLAN.md` pour voir le plan détaillé
3. **Consulter** `AI_PROMPT_TEMPLATE.md` quand vous avez besoin d'aide IA
4. **Lire** `GITHUB_WORKFLOW.md` pour la stratégie Git

### Étape 2 : Approuver le Plan

**Avant de commencer le codage**, vérifier que :
- ✅ Le plan de développement vous convient
- ✅ Vous comprenez les phases
- ✅ Vous êtes d'accord avec la stratégie Git

**Si des modifications sont nécessaires**, dites-moi et j'ajusterai.

### Étape 3 : Initialiser le Dépôt GitHub (Phase 0)

Une fois le plan approuvé, nous procéderons à :

```bash
# 1. Créer structure de dossiers
# 2. Initialiser Git
git init
git add .
git commit -m "[feat] Initial commit - Project structure"
git branch -M main
git remote add origin https://github.com/Kazaz-Mohammed/usines_intelligentes.git
git push -u origin main

# 3. Créer branche develop
git checkout -b develop
git push -u origin develop
```

## 🗺️ Vue d'Ensemble des Phases

| Phase | Objectif | Durée Estimée | Tests Requis |
|-------|----------|---------------|--------------|
| 0 | Initialisation GitHub | 1h | ✅ Structure OK |
| 1 | Infrastructure Docker | 2-3 jours | ✅ Conteneurs fonctionnent |
| 2 | IngestionIIoT | 3-4 jours | ✅ Tests unitaires + intégration |
| 3 | Prétraitement | 3-4 jours | ✅ Validation qualité données |
| 4 | ExtractionFeatures | 3-4 jours | ✅ Features correctes |
| 5 | Data Mining KNIME | 2-3 jours | ✅ Workflows exécutables |
| 6 | DétectionAnomalies | 4-5 jours | ✅ Métriques ML validées |
| 7 | PrédictionRUL | 5-7 jours | ✅ MAE < 10 cycles |
| 8 | OrchestrateurMaintenance | 3-4 jours | ✅ Règles fonctionnelles |
| 9 | DashboardUsine | 4-5 jours | ✅ Tests E2E |
| 10 | Intégration E2E | 3-4 jours | ✅ Pipeline complet |
| 11 | Kubernetes | 3-4 jours | ✅ Déploiement OK |
| 12 | Documentation | 2-3 jours | ✅ Documentation complète |

**Total estimé** : ~40-50 jours de développement

## ✅ Checklist Avant de Commencer

- [ ] J'ai lu `PROJECT_EXPLANATION.md`
- [ ] J'ai lu `DEVELOPMENT_PLAN.md`
- [ ] J'ai compris les 12 phases
- [ ] J'ai compris la stratégie Git
- [ ] J'ai accès à KNIME Analytics Platform
- [ ] J'ai Docker installé
- [ ] J'ai Java 17+ installé
- [ ] J'ai Python 3.9+ installé
- [ ] J'ai Node.js 18+ installé
- [ ] J'ai accès au dataset NASA C-MAPSS (ou je sais où le télécharger)
- [ ] Je suis prêt à suivre le plan phase par phase
- [ ] Je comprends qu'il faut tester avant de passer à la phase suivante

## 🚦 Règles Importantes

### Ne JAMAIS
- ❌ Passer à la phase suivante sans validation complète
- ❌ Push du code avec secrets/credentials
- ❌ Merge dans `main` sans validation
- ❌ Utiliser des modèles pré-entraînés pour ML/DL (contrainte module)

### Toujours
- ✅ Tester avant de push
- ✅ Push minimum 1 fois par jour
- ✅ Documenter les décisions importantes
- ✅ Créer des tags après chaque phase complète
- ✅ Utiliser messages de commit clairs

## 🎓 Utilisation de l'Assistance IA

Quand vous avez besoin d'aide pour développer :

1. **Ouvrir** `AI_PROMPT_TEMPLATE.md`
2. **Sélectionner** le prompt approprié à votre phase
3. **Adapter** selon votre besoin spécifique
4. **Copier** et utiliser avec l'IA

Exemple :
```
Je suis en Phase 2 - Service IngestionIIoT.
[Utiliser le prompt de la Phase 2 depuis AI_PROMPT_TEMPLATE.md]
```

## 📊 Comment Utiliser le Plan

### Pour Chaque Phase

1. **Lire** la section de la phase dans `DEVELOPMENT_PLAN.md`
2. **Comprendre** les objectifs et tâches
3. **Créer** la branche feature : `git checkout -b feature/[nom-service]`
4. **Développer** en suivant les tâches
5. **Tester** selon les critères de validation
6. **Valider** tous les tests passent
7. **Documenter** si nécessaire
8. **Commit et push** : `git push origin feature/[nom-service]`
9. **Créer Pull Request** ou merge dans `develop`
10. **Tag** si phase complète : `git tag v0.X.0`

### En Cas de Problème

1. **Consulter** les tests de validation de la phase
2. **Vérifier** les logs et erreurs
3. **Utiliser** le prompt de debugging depuis `AI_PROMPT_TEMPLATE.md`
4. **Si besoin** : Rollback avec Git (voir `GITHUB_WORKFLOW.md`)

## 🔄 Workflow Quotidien

### Début de Journée
```bash
# 1. Mettre à jour develop
git checkout develop
git pull origin develop

# 2. Créer/sélectionner branche feature
git checkout feature/[nom-service]
# OU
git checkout -b feature/nouvelle-feature
```

### Pendant le Développement
```bash
# Développer, tester...

# Commit fréquent
git add .
git commit -m "[feat][service] Description"
git push origin feature/[nom-service]
```

### Fin de Journée
```bash
# Push final même si incomplet
git add .
git commit -m "[feat][service] Work in progress - [date]"
git push origin feature/[nom-service]
```

## 🎯 Prochaines Étapes Après Approbation

Une fois que vous approuvez le plan :

1. **Je créerai** la structure de dossiers complète
2. **J'initialiserai** le dépôt Git avec commit initial
3. **Je configurerai** les branches (main, develop)
4. **Je créerai** les fichiers de base (.gitignore, etc.)
5. **Nous commencerons** Phase 0 : Initialisation

## ❓ Questions Fréquentes

**Q : Puis-je modifier le plan ?**
R : Oui, dites-moi ce que vous voulez changer et j'ajusterai.

**Q : Dois-je tout développer seul ?**
R : Vous pouvez utiliser l'assistance IA avec les prompts fournis.

**Q : Que faire si je bloque sur une phase ?**
R : Utilisez le prompt de debugging ou demandez de l'aide avec le contexte.

**Q : Puis-je sauter des phases ?**
R : Non recommandé. Chaque phase construit sur la précédente.

**Q : Combien de temps par jour dois-je travailler ?**
R : Selon votre planning. Minimum : push quotidien même petit.

## 📞 Support

- **Documentation** : Consulter les fichiers .md du projet
- **Plan** : `DEVELOPMENT_PLAN.md`
- **Git** : `GITHUB_WORKFLOW.md`
- **IA** : `AI_PROMPT_TEMPLATE.md`

---

## ✋ Attendre Votre Approbation

**Je n'ai pas encore commencé à coder** comme demandé.

**Attente** :
- ✅ Votre lecture et compréhension du plan
- ✅ Votre approbation ou modifications demandées
- ✅ Votre confirmation pour démarrer Phase 0

Une fois que vous êtes prêt, dites-moi et nous commencerons ! 🚀

