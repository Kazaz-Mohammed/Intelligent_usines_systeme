# Corrections Appliquées au Jenkinsfile

## Problèmes Identifiés et Corrigés

### 1. ✅ Problème Maven - Chemin avec Espaces

**Problème :** Le chemin Maven contient des espaces (`C:\Program Files\apache-maven-3.9.12`) et n'était pas entre guillemets, causant l'erreur :
```
'C:\Program' n'est pas reconnu en tant que commande interne
```

**Solution :** Ajout de guillemets autour du chemin complet :
```groovy
// Avant
bat "${mvn}\\bin\\mvn clean verify sonar:sonar ..."

// Après
bat "\"${mvn}\\bin\\mvn\" clean verify sonar:sonar ..."
```

### 2. ✅ Problème sonar-scanner - Commande Non Trouvée

**Problème :** `sonar-scanner` n'était pas trouvé dans le PATH :
```
'sonar-scanner' n'est pas reconnu en tant que commande interne
```

**Solution :** Utilisation du tool SonarQube Scanner configuré dans Jenkins :
```groovy
def scannerHome = tool 'SonarQubeScanner'
bat "\"${scannerHome}\\bin\\sonar-scanner.bat\" ..."
```

### 3. ⚠️ Erreur SonarQube UI

L'erreur "The component cannot be loaded" dans SonarQube peut être :
- Un problème temporaire (recharger la page)
- Un problème de permissions de base de données
- Les warnings dans les logs (plugins) ne sont pas critiques

**Solutions à essayer :**
1. Attendre quelques minutes et recharger la page
2. Vérifier que SonarQube est complètement démarré : `docker logs sonarqube`
3. Redémarrer SonarQube si nécessaire : `docker compose -f sonarqube-compose.yml restart sonarqube`

---

## ⚠️ Point Important : Nom du Tool SonarQube Scanner

Le nom du tool dans le Jenkinsfile est **`SonarQubeScanner`**. 

**À vérifier dans Jenkins :**
1. **Manage Jenkins** → **Tools**
2. Section **SonarQube Scanner installations**
3. Vérifier le **nom exact** du tool configuré
4. Si le nom est différent (ex: `SonarScanner`, `sonar-scanner`, etc.), il faut :
   - Soit changer le nom dans Jenkins pour qu'il corresponde
   - Soit modifier le Jenkinsfile pour utiliser le bon nom

**Pour modifier le Jenkinsfile si le nom est différent :**
Remplacez toutes les occurrences de `'SonarQubeScanner'` par le nom correct de votre tool.

---

## Prochaines Étapes

1. **Pousser le Jenkinsfile corrigé vers GitHub :**
   ```powershell
   git add Jenkinsfile
   git commit -m "fix: correct Maven path and SonarQube Scanner tool usage"
   git push origin main
   ```

2. **Dans Jenkins, relancer le build** ou attendre le déclenchement automatique

3. **Vérifier les logs du build** pour confirmer que les erreurs sont corrigées

4. **Si l'erreur SonarQube UI persiste :**
   - Vérifier les logs : `docker logs sonarqube`
   - Redémarrer SonarQube si nécessaire
   - Vérifier que la base de données PostgreSQL est accessible

