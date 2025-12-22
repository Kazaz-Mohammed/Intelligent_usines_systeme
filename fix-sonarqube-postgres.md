# Fix pour SonarQube PostgreSQL 18+ Error

## Problème

PostgreSQL 18+ nécessite un changement dans la configuration du volume. L'erreur indique :
```
Error: in 18+, these Docker images are configured to store database data in a
format which is compatible with "pg_ctlcluster" (specifically, using
major-version-specific directory names).

Counter to that, there appears to be PostgreSQL data in:
  /var/lib/postgresql/data (unused mount/volume)
```

## Solution

Le volume doit être monté sur `/var/lib/postgresql` au lieu de `/var/lib/postgresql/data`.

## Correction Appliquée

Le fichier `sonarqube-compose.yml` a été corrigé. Le volume PostgreSQL est maintenant monté sur `/var/lib/postgresql`.

## Si vous aviez des données existantes

Si vous aviez des données dans l'ancien volume, vous devez les supprimer et redémarrer :

```powershell
# Arrêter les containers
docker compose -f sonarqube-compose.yml down

# Supprimer l'ancien volume (si vous n'avez pas besoin des données)
docker volume rm usines_intelligentes_new_postgres_data

# Redémarrer avec la nouvelle configuration
docker compose -f sonarqube-compose.yml up -d
```

⚠️ **Attention** : Supprimer le volume supprimera toutes les données existantes (projets SonarQube, analyses, etc.).

## Redémarrer SonarQube

```powershell
docker compose -f sonarqube-compose.yml up -d
```

Vérifier que tout démarre correctement :
```powershell
docker ps | findstr sonarqube
docker logs sonarqube-db
```

