# Guide PostgreSQL - Service Prediction RUL

## Vue d'ensemble

Le service `prediction-rul` utilise PostgreSQL pour journaliser toutes les prédictions RUL, permettant l'historique, l'analyse et le reporting.

## ⚠️ Instructions obligatoires

### 1. Créer la base de données PostgreSQL (OBLIGATOIRE)

**Tu dois exécuter ces commandes SQL une seule fois** pour créer la base de données et l'utilisateur :

```sql
CREATE DATABASE predictive_maintenance;
CREATE USER pmuser WITH PASSWORD 'pmpassword';
GRANT ALL PRIVILEGES ON DATABASE predictive_maintenance TO pmuser;
```

**Comment faire :**
- Connecte-toi à PostgreSQL (via `psql` ou un client graphique)
- Exécute les 3 commandes SQL ci-dessus
- C'est tout ! Le service créera automatiquement les tables au démarrage

### 2. Configurer les variables d'environnement (OBLIGATOIRE)

**Tu dois configurer ces variables** dans ton fichier `.env` ou comme variables d'environnement :

```bash
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=predictive_maintenance
DATABASE_USER=pmuser
DATABASE_PASSWORD=pmpassword
```

**C'est tout pour la configuration obligatoire !** Le service créera automatiquement les tables et index au démarrage.

## 📋 Instructions optionnelles (pour référence)

Les sections suivantes sont des **exemples et guides de référence** - tu n'as pas besoin de les exécuter maintenant :

## Structure de la table (créée automatiquement)

**⚠️ Tu n'as RIEN à faire ici** - La table `rul_predictions` est créée automatiquement au démarrage du service :

```sql
CREATE TABLE rul_predictions (
    id SERIAL PRIMARY KEY,
    asset_id VARCHAR(255) NOT NULL,
    sensor_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    rul_prediction DECIMAL(10, 2) NOT NULL,
    confidence_interval_lower DECIMAL(10, 2),
    confidence_interval_upper DECIMAL(10, 2),
    confidence_level DECIMAL(3, 2) DEFAULT 0.95,
    uncertainty DECIMAL(10, 2),
    model_used VARCHAR(50) NOT NULL,
    model_scores JSONB,
    features JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### Index

Les index suivants sont créés automatiquement pour optimiser les requêtes :

- `idx_rul_asset_id` - Sur `asset_id`
- `idx_rul_sensor_id` - Sur `sensor_id`
- `idx_rul_timestamp` - Sur `timestamp`
- `idx_rul_model_used` - Sur `model_used`
- `idx_rul_asset_timestamp` - Composite sur `(asset_id, timestamp DESC)`
- `idx_rul_sensor_timestamp` - Composite sur `(sensor_id, timestamp DESC)`

## Journalisation automatique (fonctionne automatiquement)

**⚠️ Tu n'as RIEN à faire ici** - La journalisation fonctionne automatiquement :

- **Via l'API REST** : Les prédictions RUL via `POST /api/v1/rul/predict` sont automatiquement journalisées
- **Via le Worker Kafka** : Le worker Kafka journalise automatiquement toutes les prédictions RUL en temps-réel

## API REST

### GET /api/v1/rul/

Récupère l'historique des prédictions RUL avec filtres et pagination :

```bash
# Toutes les prédictions
curl http://localhost:8085/api/v1/rul/

# Filtrer par asset_id
curl "http://localhost:8085/api/v1/rul/?asset_id=ASSET001"

# Filtrer par modèle utilisé
curl "http://localhost:8085/api/v1/rul/?model_used=lstm"

# Avec pagination
curl "http://localhost:8085/api/v1/rul/?limit=50&offset=0"

# Avec dates
curl "http://localhost:8085/api/v1/rul/?start_date=2024-01-01T00:00:00Z&end_date=2024-01-31T23:59:59Z"
```

**Réponse :**
```json
{
    "predictions": [
        {
            "id": 1,
            "asset_id": "ASSET001",
            "sensor_id": "SENSOR001",
            "timestamp": "2024-01-15T10:30:00Z",
            "rul_prediction": 150.5,
            "confidence_interval_lower": 140.0,
            "confidence_interval_upper": 160.0,
            "confidence_level": 0.95,
            "uncertainty": 10.0,
            "model_used": "ensemble",
            "model_scores": {"lstm": 150.0, "gru": 151.0},
            "features": {"rms": 10.5, "kurtosis": 2.3},
            "metadata": {}
        }
    ],
    "total": 1,
    "limit": 100,
    "offset": 0,
    "filters": {
        "asset_id": null,
        "sensor_id": null,
        "start_date": null,
        "end_date": null,
        "model_used": null
    }
}
```

## Requêtes SQL utiles

### Compter les prédictions par actif

```sql
SELECT asset_id, COUNT(*) as prediction_count
FROM rul_predictions
GROUP BY asset_id
ORDER BY prediction_count DESC;
```

### Dernière prédiction pour chaque actif

```sql
SELECT DISTINCT ON (asset_id)
    asset_id, rul_prediction, timestamp, model_used
FROM rul_predictions
ORDER BY asset_id, timestamp DESC;
```

### Prédictions avec faible RUL (< 50 cycles)

```sql
SELECT asset_id, rul_prediction, timestamp, model_used
FROM rul_predictions
WHERE rul_prediction < 50
ORDER BY rul_prediction ASC, timestamp DESC;
```

### Évolution de la RUL pour un actif

```sql
SELECT timestamp, rul_prediction, confidence_interval_lower, confidence_interval_upper
FROM rul_predictions
WHERE asset_id = 'ASSET001'
ORDER BY timestamp ASC;
```

## Performance

### Optimisations automatiques

- **Index automatiques** : Tous les index nécessaires sont créés automatiquement
- **Pool de connexions** : Gestion automatique du pool (1-10 connexions)
- **JSONB** : Utilisation de JSONB pour les champs complexes (scores, features, metadata)

### Recommandations

- **Archivage** : Considérer l'archivage des anciennes prédictions (> 1 an)
- **Partitioning** : Pour de très grandes tables, considérer le partitioning par date
- **TimescaleDB** : Pour des performances optimales avec des données temporelles, considérer TimescaleDB

## Troubleshooting

### Erreur de connexion

```
Erreur lors de l'initialisation du pool PostgreSQL: ...
```

**Solution :**
1. Vérifier que PostgreSQL est démarré
2. Vérifier les credentials dans `.env`
3. Vérifier que la base de données existe

### Table n'existe pas

```
relation "rul_predictions" does not exist
```

**Solution :**
- Le service devrait créer la table automatiquement au démarrage
- Vérifier les logs pour les erreurs de création
- Vérifier les permissions de l'utilisateur PostgreSQL

### Erreur de journalisation

Si la journalisation échoue, le service continue de fonctionner (les prédictions sont toujours retournées, mais non journalisées).

**Vérifier les logs :**
```bash
# Chercher les erreurs PostgreSQL dans les logs
grep -i "postgresql\|database" logs/app.log
```

## Maintenance

### Sauvegarde

```bash
# Sauvegarder la base de données
pg_dump -U pmuser -d predictive_maintenance > backup.sql

# Restaurer
psql -U pmuser -d predictive_maintenance < backup.sql
```

### Nettoyage des anciennes données

```sql
-- Supprimer les prédictions de plus d'1 an
DELETE FROM rul_predictions
WHERE timestamp < NOW() - INTERVAL '1 year';
```

### Statistiques

```sql
-- Statistiques générales
SELECT 
    COUNT(*) as total_predictions,
    COUNT(DISTINCT asset_id) as unique_assets,
    AVG(rul_prediction) as avg_rul,
    MIN(rul_prediction) as min_rul,
    MAX(rul_prediction) as max_rul
FROM rul_predictions;
```

