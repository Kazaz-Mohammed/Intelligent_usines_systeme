# Dataset NASA C-MAPSS

## Description

Dataset NASA Commercial Modular Aero-Propulsion System Simulation utilisé pour l'entraînement des modèles RUL.

## Caractéristiques

- **21 capteurs** de surveillance
- **3 réglages** de moteur
- **4 scénarios** de dégradation différents
- Format CSV

## Téléchargement

Le dataset doit être téléchargé depuis :
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pc-pda-data-set-repository/

## Structure Attendue

```
nasa-cmapss/
├── train_FD001.txt
├── train_FD002.txt
├── train_FD003.txt
├── train_FD004.txt
├── test_FD001.txt
├── test_FD002.txt
├── test_FD003.txt
└── test_FD004.txt
```

## Notes

- Les fichiers sont volumineux, ne pas les committer sur Git
- Utiliser `.gitignore` pour exclure les fichiers de données

