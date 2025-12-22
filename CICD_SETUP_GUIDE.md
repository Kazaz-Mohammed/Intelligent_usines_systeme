# Guide de Configuration CI/CD - Pipeline Jenkins + SonarQube + Ngrok

## Vue d'Ensemble

Ce guide vous accompagne dans la mise en place d'un pipeline CI/CD pour votre projet de maintenance prédictive.

**Mode Actuel : CI-Only (Pas de Déploiement)**
- ✅ Build automatique des services
- ✅ Analyse de code avec SonarQube
- ❌ Pas de déploiement automatique (vos containers existants ne seront pas touchés)

---

## 📋 État Actuel

- ✅ **Jenkins** : Installé
- ⚠️ **SonarQube** : Containers existants mais arrêtés
- ✅ **Ngrok** : Configuré et actif
  - URL : `https://amalia-proterogynous-subangularly.ngrok-free.dev`
  - Port : 80
  - ⚠️ **Important** : Vérifiez que Jenkins est bien sur le port 80, sinon relancez ngrok avec le bon port (généralement 8080)
- ✅ **GitHub** : https://github.com/Kazaz-Mohammed/Intelligent_usines_systeme.git

---

## 🔧 Étape 1 : Démarrer SonarQube

### 1.1 Démarrer les containers SonarQube

```powershell
# Depuis la racine du projet
docker compose -f sonarqube-compose.yml up -d
```

### 1.2 Vérifier que SonarQube démarre

```powershell
docker ps | findstr sonarqube
```

Attendez 1-2 minutes que SonarQube soit complètement démarré (premier démarrage peut prendre du temps).

### 1.3 Accéder à SonarQube

1. Ouvrez votre navigateur : http://localhost:9999
2. Login par défaut :
   - Username : `admin`
   - Password : `admin`
3. **Changez le mot de passe** lors du premier login (important!)

---

## 🔧 Étape 2 : Configurer SonarQube dans Jenkins

### 2.1 Installer le Plugin SonarQube Scanner

1. Ouvrez Jenkins : `http://localhost:8080` (ou le port configuré)
2. **Manage Jenkins** → **Plugins** → **Available plugins**
3. Recherchez et installez :
   - **SonarQube Scanner for Jenkins** (si pas déjà installé)
   - **Pipeline** (si pas déjà installé)
4. Redémarrez Jenkins si demandé

### 2.2 Configurer SonarQube Scanner dans Jenkins Tools

1. **Manage Jenkins** → **Tools**
2. Section **SonarQube Scanner installations**
3. Cliquez **Add SonarQube Scanner**
4. Options :
   - **Name** : `SonarQubeScanner` (ou autre nom cohérent)
   - **Install automatically** : ✅ Cocher
   - Choisir une version (ex: latest)
5. **Save**

### 2.3 Configurer le Serveur SonarQube dans Jenkins System

1. **Manage Jenkins** → **System**
2. Faites défiler jusqu'à la section **SonarQube servers**
3. Cliquez **Add SonarQube**
4. Remplissez :
   - **Name** : `SonarQube` ⚠️ **Important** : Ce nom doit correspondre exactement à celui dans le Jenkinsfile
   - **Server URL** : `http://localhost:9999`
   - **Server authentication token** : (générer depuis SonarQube - voir étape suivante)

### 2.4 Générer un Token SonarQube

1. Dans SonarQube (http://localhost:9999), cliquez sur votre avatar (en haut à droite)
2. **My Account** → **Security**
3. Section **Generate Tokens**
4. **Name** : `jenkins-token`
5. **Type** : `User Token`
6. Cliquez **Generate**
7. **⚠️ COPIEZ LE TOKEN** (il ne sera plus visible après!)
8. Collez ce token dans Jenkins (Étape 2.3, champ "Server authentication token")
9. **Save** dans Jenkins

---

## 🔧 Étape 3 : Créer les Projets SonarQube

Pour chaque service, créez un projet dans SonarQube :

### 3.1 Services à créer

Créer les projets suivants dans SonarQube :

1. **ingestion-iiot**
2. **orchestrateur-maintenance**
3. **preprocessing**
4. **extraction-features**
5. **detection-anomalies**
6. **prediction-rul**

### 3.2 Créer un projet

1. Dans SonarQube, **Projects** → **Create Project**
2. Sélectionnez **Manually**
3. **Project key** : `ingestion-iiot` (exemple)
4. **Display name** : `IngestionIIoT Service`
5. **Main branch name** : `main`
6. Cliquez **Set Up**

**Répétez pour chaque service** (ou créez-les lors de la première analyse - SonarQube peut créer automatiquement).

---

## 🔧 Étape 4 : Vérifier Ngrok et Port Jenkins

### 4.1 Vérifier le port de Jenkins

Jenkins est généralement sur le port **8080**, pas 80.

Vérifiez dans votre installation Jenkins quel port est utilisé.

### 4.2 Relancer Ngrok avec le bon port

Si Jenkins est sur le port 8080 :

```powershell
# Arrêter le ngrok actuel (Ctrl+C dans la fenêtre ngrok)
# Relancer avec le port Jenkins
ngrok http 8080
```

**Notez la nouvelle URL Ngrok** (ex: `https://nouvelle-url.ngrok-free.app`)

⚠️ **Important** : L'URL Ngrok change à chaque redémarrage. Vous devrez mettre à jour le webhook GitHub si elle change.

### 4.3 URL Ngrok actuelle

Votre URL Ngrok actuelle : `https://amalia-proterogynous-subangularly.ngrok-free.dev`

**Si Jenkins est sur le port 80** : Gardez cette URL  
**Si Jenkins est sur le port 8080** : Relancez ngrok avec `ngrok http 8080` et utilisez la nouvelle URL

---

## 🔧 Étape 5 : Configurer GitHub dans Jenkins

### 5.1 Installer les Plugins GitHub (si pas déjà fait)

1. **Manage Jenkins** → **Plugins** → **Available plugins**
2. Recherchez et installez :
   - **GitHub plugin**
   - **GitHub Integration plugin**
   - **Git plugin** (généralement déjà installé)
3. Redémarrez Jenkins si demandé

### 5.2 Configurer GitHub dans Jenkins System

1. **Manage Jenkins** → **System**
2. Section **GitHub**
3. **GitHub Pull Requests** :
   - **Published Jenkins URL** : `https://<VOTRE_URL_NGROK>` (sans le `/` final)
   - Exemple : `https://amalia-proterogynous-subangularly.ngrok-free.dev`
4. **Project url** : `https://github.com/Kazaz-Mohammed/Intelligent_usines_systeme`
5. **Save**

---

## 🔧 Étape 6 : Configurer Maven dans Jenkins

### 6.1 Vérifier/Créer l'installation Maven

1. **Manage Jenkins** → **Tools**
2. Section **Maven installations**
3. Si aucune installation n'existe, cliquez **Add Maven**
4. Remplissez :
   - **Name** : `maven` ⚠️ **Important** : Le nom doit être exactement `maven` (comme dans le Jenkinsfile)
   - **MAVEN_HOME** : Chemin vers votre installation Maven locale
   - OU cochez **Install automatically** et choisissez une version
5. **Save**

---

## 🔧 Étape 7 : Créer le Job Pipeline Jenkins

### 7.1 Créer un nouveau Job

1. Dans Jenkins, **Dashboard** → **New Item**
2. **Item name** : `cicd-microservices-pipeline` (ou autre nom)
3. Sélectionnez **Pipeline**
4. Cliquez **OK**

### 7.2 Configurer le Job

#### Onglet **General**

1. ✅ Cocher **GitHub project**
2. **Project url** : `https://github.com/Kazaz-Mohammed/Intelligent_usines_systeme`

#### Onglet **Build Triggers**

1. ✅ Cocher **GitHub hook trigger for GITScm polling**

#### Onglet **Pipeline**

1. **Definition** : **Pipeline script from SCM** (recommandé) OU **Pipeline script** (si vous copiez le script directement)

**Option A : Pipeline script from SCM** (recommandé)
- **SCM** : `Git`
- **Repository URL** : `https://github.com/Kazaz-Mohammed/Intelligent_usines_systeme.git`
- **Credentials** : Aucun (si repo public) ou ajouter vos credentials GitHub
- **Branches to build** : `*/main`
- **Script Path** : `Jenkinsfile`

**Option B : Pipeline script** (copier-coller)
- Collez le contenu du fichier `Jenkinsfile` (créé dans ce projet)
- ⚠️ Moins pratique car nécessite de mettre à jour manuellement dans Jenkins

2. Cliquez **Save**

---

## 🔧 Étape 8 : Créer le Webhook GitHub

### 8.1 Accéder aux Webhooks GitHub

1. Allez sur votre repo GitHub : https://github.com/Kazaz-Mohammed/Intelligent_usines_systeme
2. **Settings** → **Webhooks** → **Add webhook**

### 8.2 Configurer le Webhook

1. **Payload URL** : `https://<VOTRE_URL_NGROK>/github-webhook/`
   - Exemple : `https://amalia-proterogynous-subangularly.ngrok-free.dev/github-webhook/`
   - ⚠️ Important : Le `/` final est important!
2. **Content type** : `application/json`
3. **Which events** : **Just the push event** (ou "Send me everything" pour tester)
4. ✅ **Active** : Cocher
5. Cliquez **Add webhook**

### 8.3 Vérifier le Webhook

1. Après création, GitHub tentera d'envoyer un "ping"
2. Vérifiez que le statut est **200 OK** (ou vert)
3. Si erreur, vérifiez l'URL Ngrok et que Jenkins est accessible

---

## 🔧 Étape 9 : Premier Test - Build Manuel

### 9.1 Lancer un Build Manuel

1. Dans Jenkins, ouvrez votre job `cicd-microservices-pipeline`
2. Cliquez **Build Now**
3. Attendez la fin du build (peut prendre plusieurs minutes)

### 9.2 Vérifier les Résultats

1. Cliquez sur le build dans l'historique
2. Cliquez **Console Output** pour voir les logs
3. Vérifiez que :
   - ✅ Le clonage GitHub fonctionne
   - ✅ Les builds Maven passent (services Java)
   - ✅ Les installations Python passent
   - ✅ Les analyses SonarQube s'exécutent

### 9.3 Vérifier SonarQube

1. Allez sur http://localhost:9999
2. **Projects** → Vérifiez que les projets apparaissent avec des analyses récentes
3. Cliquez sur un projet pour voir les métriques (bugs, vulnérabilités, code smells)

---

## 🔧 Étape 10 : Tester le Déclenchement Automatique

### 10.1 Faire un Push de Test

```powershell
# Depuis votre projet local
git add .
git commit -m "test: déclenchement webhook CI/CD"
git push origin main
```

### 10.2 Vérifier dans Jenkins

1. Dans Jenkins, ouvrez votre job
2. Un nouveau build devrait démarrer automatiquement (quelques secondes après le push)
3. Vérifiez la console output

---

## ✅ Checklist Finale

### Configuration
- [ ] SonarQube démarré et accessible (http://localhost:9999)
- [ ] Token SonarQube généré et configuré dans Jenkins
- [ ] Serveur SonarQube configuré dans Jenkins System (nom: "SonarQube")
- [ ] SonarQube Scanner configuré dans Jenkins Tools
- [ ] Maven configuré dans Jenkins Tools (nom: "maven")
- [ ] Plugins Jenkins installés (SonarQube, GitHub, Pipeline)

### GitHub et Ngrok
- [ ] Ngrok actif et pointant vers le bon port Jenkins
- [ ] URL Ngrok notée
- [ ] GitHub configuré dans Jenkins System
- [ ] Webhook GitHub créé avec l'URL Ngrok
- [ ] Webhook GitHub testé (statut 200 OK)

### Pipeline
- [ ] Job Pipeline créé dans Jenkins
- [ ] Jenkinsfile présent dans le repo GitHub
- [ ] Build manuel réussi
- [ ] Déclenchement automatique testé (push GitHub)

---

## 🐛 Dépannage

### SonarQube inaccessible
```powershell
docker ps | findstr sonarqube
docker logs sonarqube  # Voir les logs
docker compose -f sonarqube-compose.yml up -d  # Redémarrer
```

### Build Jenkins échoue - "maven not found"
- Vérifier que Maven est configuré dans Jenkins Tools avec le nom exact "maven"

### Build Jenkins échoue - "SonarQube not found"
- Vérifier que le serveur SonarQube est configuré dans Jenkins System avec le nom exact "SonarQube"
- Vérifier que le token est correct

### Webhook GitHub échoue (404)
- Vérifier l'URL Ngrok (elle change à chaque redémarrage)
- Vérifier que le `/github-webhook/` est présent à la fin de l'URL
- Vérifier que Jenkins est accessible via l'URL Ngrok

### Python build échoue
- Vérifier que Python est dans le PATH de Jenkins
- Vérifier les versions Python (le pipeline utilise Python 3.9)

### SonarQube Scanner non trouvé (services Python)
- Vérifier que SonarQube Scanner est installé dans Jenkins Tools
- Ou installer sonar-scanner localement et l'ajouter au PATH

---

## 📝 Notes Importantes

1. **Mode CI-Only** : Le pipeline actuel ne déploie PAS automatiquement. Vos containers existants ne seront pas affectés.

2. **URL Ngrok** : L'URL change à chaque redémarrage de ngrok. Mettez à jour le webhook GitHub si nécessaire.

3. **SonarQube** : Premier démarrage peut prendre 1-2 minutes. Les analyses peuvent prendre plusieurs minutes.

4. **Build Parallèle** : Les services sont construits en parallèle pour optimiser le temps d'exécution.

5. **Déploiement Futur** : Pour activer le déploiement, décommentez le stage "Docker Compose" dans le Jenkinsfile et créez `deploy/docker-compose.yml`.

---

## 🚀 Prochaines Étapes (Optionnel)

Une fois le CI fonctionnel, vous pourrez :

1. **Ajouter des tests automatiques** dans le pipeline
2. **Ajouter des notifications** (email, Slack) en cas d'échec
3. **Activer le CD** (déploiement automatique) en créant `deploy/docker-compose.yml`
4. **Configurer des environnements** (dev, staging, prod)
5. **Ajouter le build/push d'images Docker** vers un registry

---

**Dernière mise à jour** : Janvier 2025

