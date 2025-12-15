# Guide de Contribution

## Workflow de Développement

Ce projet suit un workflow Git strict pour garantir la qualité du code.

### Branches

- `main` : Code production-ready uniquement
- `develop` : Branche de développement principale
- `feature/[nom]` : Nouvelles fonctionnalités

### Commits

Utiliser la convention :
```
[TYPE][SERVICE] Description

Exemples:
[feat][ingestion-iiot] Ajout support OPC UA
[fix][preprocessing] Correction rééchantillonnage
```

### Pull Requests

1. Créer une branche depuis `develop`
2. Développer et tester
3. Créer Pull Request vers `develop`
4. Attendre validation avant merge

Voir `GITHUB_WORKFLOW.md` pour plus de détails.

