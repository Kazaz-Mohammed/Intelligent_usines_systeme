# Résolution du Problème SonarQube

## Problème Identifié

1. **SonarQube utilisait H2 au lieu de PostgreSQL** : Les logs montraient "Starting embedded database on port 9092 with url jdbc:h2" au lieu de PostgreSQL.

2. **Page de changement de mot de passe qui charge indéfiniment** : Probablement lié à l'utilisation de H2 ou à l'initialisation en cours.

## Causes

1. **Mauvais nom de variable d'environnement** : `SONARQUBE_JDBC_URL` au lieu de `SONAR_JDBC_URL`
2. **Volumes existants avec données H2** : SonarQube avait déjà initialisé une base H2 dans les volumes

## Solutions Appliquées

### 1. Correction des Variables d'Environnement

**Avant :**
```yaml
environment:
  - SONARQUBE_JDBC_URL=jdbc:postgresql://sonarqube-db:5432/sonarqube
```

**Après :**
```yaml
environment:
  - SONAR_JDBC_URL=jdbc:postgresql://sonarqube-db:5432/sonarqube
```

### 2. Ajout d'un Healthcheck PostgreSQL

Ajout d'un healthcheck pour s'assurer que PostgreSQL est prêt avant que SonarQube démarre :

```yaml
sonarqube-db:
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U sonar -d sonarqube"]
    interval: 10s
    timeout: 5s
    retries: 5
```

### 3. Suppression des Volumes

Suppression de tous les volumes pour repartir à zéro avec PostgreSQL :

```powershell
docker compose -f sonarqube-compose.yml down -v
docker compose -f sonarqube-compose.yml up -d
```

## Résultat

✅ SonarQube utilise maintenant PostgreSQL (visible dans les logs : `jdbc:postgresql://sonarqube-db:5432/sonarqube`)

✅ PostgreSQL est accessible et fonctionnel

## Prochaines Étapes

1. **Attendre l'initialisation complète** : SonarQube peut prendre 1-2 minutes pour terminer son initialisation après le premier démarrage avec PostgreSQL.

2. **Accéder à SonarQube** : http://localhost:9999

3. **Se connecter** :
   - Username : `admin`
   - Password : `admin`
   - Changer le mot de passe au premier login

4. **Si la page charge encore** : Attendre 1-2 minutes supplémentaires, puis actualiser la page. L'initialisation de la base de données PostgreSQL peut prendre du temps.

## Vérification

Pour vérifier que SonarQube utilise PostgreSQL :

```powershell
docker logs sonarqube | Select-String -Pattern "postgresql|PostgreSQL"
```

Vous devriez voir : `Create JDBC data source for jdbc:postgresql://sonarqube-db:5432/sonarqube`

**Plus de mention de H2 dans les logs** ✅

