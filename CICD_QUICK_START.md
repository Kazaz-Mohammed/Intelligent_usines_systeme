# CI/CD Quick Start - Actions Manuelles Requises

## ⚡ Actions Immédiates à Faire

### 1. ⚠️ Vérifier le Port Jenkins et Relancer Ngrok

Votre Ngrok pointe actuellement vers le port **80**, mais Jenkins est généralement sur le port **8080**.

**À faire :**
```powershell
# Vérifier quel port Jenkins utilise (généralement 8080)
# Puis relancer ngrok avec le bon port :
ngrok http 8080
```

**Notez la nouvelle URL Ngrok** (elle changera si vous relancez ngrok)

---

### 2. Démarrer SonarQube

```powershell
# Depuis la racine du projet
docker compose -f sonarqube-compose.yml up -d

# Vérifier que ça démarre (attendre 1-2 minutes)
docker ps | findstr sonarqube
```

Accéder à : http://localhost:9999
- Login : `admin` / `admin`
- **Changez le mot de passe** au premier login

---

### 3. Configurer SonarQube dans Jenkins (Actions Manuelles)

#### 3.1 Installer le Plugin SonarQube Scanner
1. Jenkins → **Manage Jenkins** → **Plugins** → **Available plugins**
2. Rechercher : **SonarQube Scanner for Jenkins**
3. Installer et redémarrer Jenkins

#### 3.2 Configurer SonarQube Scanner Tool
1. **Manage Jenkins** → **Tools**
2. **SonarQube Scanner installations** → **Add SonarQube Scanner**
3. **Name** : `SonarQubeScanner` (ou laisser le nom par défaut)
4. **Install automatically** : ✅ Cocher
5. **Save**

#### 3.3 Configurer le Serveur SonarQube
1. **Manage Jenkins** → **System**
2. Section **SonarQube servers** → **Add SonarQube**
3. Remplir :
   - **Name** : `SonarQube` ⚠️ **Doit être exactement "SonarQube"**
   - **Server URL** : `http://localhost:9999`
   - **Server authentication token** : Générer depuis SonarQube (voir ci-dessous)

#### 3.4 Générer Token SonarQube
1. SonarQube (http://localhost:9999) → Avatar (en haut droite) → **My Account** → **Security**
2. Section **Generate Tokens**
3. **Name** : `jenkins-token`
4. **Generate** → **Copier le token**
5. Coller dans Jenkins (étape 3.3)

---

### 4. Configurer Maven dans Jenkins

1. **Manage Jenkins** → **Tools**
2. **Maven installations** → **Add Maven**
3. **Name** : `maven` ⚠️ **Doit être exactement "maven"**
4. **MAVEN_HOME** : Chemin vers votre Maven local
   - OU cocher **Install automatically**
5. **Save**

---

### 5. Configurer GitHub dans Jenkins

#### 5.1 Installer Plugins GitHub (si nécessaire)
1. **Manage Jenkins** → **Plugins** → **Available plugins**
2. Installer : **GitHub plugin**, **GitHub Integration plugin**
3. Redémarrer Jenkins

#### 5.2 Configurer GitHub System
1. **Manage Jenkins** → **System**
2. Section **GitHub**
3. **Published Jenkins URL** : `https://<VOTRE_URL_NGROK>` (sans `/` final)
4. **Project url** : `https://github.com/Kazaz-Mohammed/Intelligent_usines_systeme`
5. **Save**

---

### 6. Créer le Job Pipeline dans Jenkins

1. Jenkins Dashboard → **New Item**
2. **Item name** : `cicd-microservices-pipeline`
3. Type : **Pipeline** → **OK**

#### Configuration du Job :

**General :**
- ✅ Cocher **GitHub project**
- **Project url** : `https://github.com/Kazaz-Mohammed/Intelligent_usines_systeme`

**Build Triggers :**
- ✅ Cocher **GitHub hook trigger for GITScm polling**

**Pipeline :**
- **Definition** : **Pipeline script from SCM**
- **SCM** : `Git`
- **Repository URL** : `https://github.com/Kazaz-Mohammed/Intelligent_usines_systeme.git`
- **Branches to build** : `*/main`
- **Script Path** : `Jenkinsfile`
- **Save**

---

### 7. Créer le Webhook GitHub

1. GitHub → https://github.com/Kazaz-Mohammed/Intelligent_usines_systeme
2. **Settings** → **Webhooks** → **Add webhook**
3. Remplir :
   - **Payload URL** : `https://<VOTRE_URL_NGROK>/github-webhook/`
     - Exemple : `https://amalia-proterogynous-subangularly.ngrok-free.dev/github-webhook/`
     - ⚠️ Le `/` final est important!
   - **Content type** : `application/json`
   - **Which events** : **Just the push event**
   - ✅ **Active** : Cocher
4. **Add webhook**
5. Vérifier le statut (doit être vert/200 OK)

---

### 8. Pousser le Code vers GitHub

```powershell
# Vérifier que vous êtes sur la bonne branche
git branch

# Ajouter tous les nouveaux fichiers
git add .

# Commit
git commit -m "feat: add CI/CD pipeline configuration (CI-only mode)"

# Pousser vers GitHub
git push origin main
```

⚠️ **Important** : Assurez-vous que le fichier `Jenkinsfile` est bien dans le repo GitHub.

---

### 9. Tester le Pipeline

#### Test Manuel :
1. Jenkins → Ouvrir le job `cicd-microservices-pipeline`
2. **Build Now**
3. Attendre la fin (peut prendre 5-10 minutes)
4. Vérifier la **Console Output**

#### Test Automatique :
1. Faire un petit changement dans le code
2. Commit et push :
   ```powershell
   git add .
   git commit -m "test: trigger CI pipeline"
   git push origin main
   ```
3. Vérifier dans Jenkins qu'un nouveau build démarre automatiquement

---

## ✅ Checklist Rapide

- [ ] Ngrok relancé avec le bon port Jenkins (probablement 8080)
- [ ] URL Ngrok notée
- [ ] SonarQube démarré (http://localhost:9999)
- [ ] Token SonarQube généré et configuré dans Jenkins
- [ ] Serveur SonarQube configuré dans Jenkins (nom: "SonarQube")
- [ ] SonarQube Scanner configuré dans Jenkins Tools
- [ ] Maven configuré dans Jenkins Tools (nom: "maven")
- [ ] Plugins installés (SonarQube Scanner, GitHub)
- [ ] GitHub configuré dans Jenkins System
- [ ] Job Pipeline créé dans Jenkins
- [ ] Webhook GitHub créé avec l'URL Ngrok
- [ ] Code poussé vers GitHub (avec Jenkinsfile)
- [ ] Build manuel testé dans Jenkins
- [ ] Déclenchement automatique testé (push)

---

## 📚 Documentation Complète

Pour plus de détails, voir : **[CICD_SETUP_GUIDE.md](CICD_SETUP_GUIDE.md)**

---

## 🐛 Problèmes Courants

### "maven not found"
→ Vérifier que Maven est configuré dans Jenkins Tools avec le nom exact "maven"

### "SonarQube not found"
→ Vérifier que le serveur SonarQube est configuré avec le nom exact "SonarQube" dans Jenkins System

### Webhook 404
→ Vérifier l'URL Ngrok et que Jenkins est accessible via cette URL

### SonarQube inaccessible
→ Vérifier : `docker ps | findstr sonarqube` et `docker logs sonarqube`

---

**Bon courage ! 🚀**

