# Déploiement Docker Compose

Ce répertoire contient la configuration Docker Compose pour le déploiement via CI/CD.

## ⚠️ Note Importante

**Actuellement, le pipeline CI/CD est configuré en mode CI-only (pas de déploiement).**

Ce répertoire est préparé pour une utilisation future lorsque vous activerez le stage de déploiement dans le pipeline Jenkins.

## Structure

```
deploy/
├── docker-compose.yml  (à créer quand vous activez le CD)
└── README.md
```

## Activation du Déploiement

Pour activer le déploiement automatique via Jenkins :

1. Créer `deploy/docker-compose.yml` (copie adaptée du docker-compose.yml principal)
2. Décommenter le stage "Docker Compose - Déploiement" dans le Jenkinsfile
3. Vérifier que Jenkins a les permissions Docker nécessaires

## Pour l'instant

Ce répertoire est vide et n'affecte pas le fonctionnement actuel du projet.

