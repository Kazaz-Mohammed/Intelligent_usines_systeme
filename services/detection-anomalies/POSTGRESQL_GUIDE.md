# Guide PostgreSQL - Service Detection Anomalies

## Vue d'ensemble

Le service `detection-anomalies` utilise PostgreSQL pour journaliser toutes les anomalies détectées, permettant l'historique, l'analyse et le reporting.

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

**⚠️ Tu n'as RIEN à faire ici** - La table `anomaly_detections` est créée automatiquement au démarrage du service :

```sql
CREATE TABLE anomaly_detections (
    id SERIAL PRIMARY KEY,
    asset_id VARCHAR(255) NOT NULL,
    sensor_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    final_score DECIMAL(5, 4) NOT NULL,
    is_anomaly BOOLEAN NOT NULL,
    criticality VARCHAR(20) NOT NULL,
    scores JSONB NOT NULL,
    features JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### Index

Les index suivants sont créés automatiquement pour optimiser les requêtes :

- `idx_asset_id` - Sur `asset_id`
- `idx_sensor_id` - Sur `sensor_id`
- `idx_timestamp` - Sur `timestamp`
- `idx_is_anomaly` - Sur `is_anomaly`
- `idx_criticality` - Sur `criticality`
- `idx_asset_timestamp` - Composite sur `(asset_id, timestamp DESC)`
- `idx_sensor_timestamp` - Composite sur `(sensor_id, timestamp DESC)`

## Journalisation automatique (fonctionne automatiquement)

**⚠️ Tu n'as RIEN à faire ici** - La journalisation fonctionne automatiquement :

- **Via l'API REST** : Les anomalies détectées via `POST /api/v1/anomalies/detect` sont automatiquement journalisées si `is_anomaly=True`
- **Via le Worker Kafka** : Le worker Kafka journalise automatiquement toutes les anomalies détectées en temps-réel

## 📖 Utilisation de l'API (exemples)

**Ces exemples montrent comment utiliser l'API** - tu peux les tester quand tu veux :

#### Toutes les anomalies

```bash
curl "http://localhost:8084/api/v1/anomalies/"
```

#### Filtrer par asset_id

```bash
curl "http://localhost:8084/api/v1/anomalies/?asset_id=ASSET001"
```

#### Filtrer par sensor_id

```bash
curl "http://localhost:8084/api/v1/anomalies/?sensor_id=SENSOR001"
```

#### Filtrer par dates

```bash
curl "http://localhost:8084/api/v1/anomalies/?start_date=2024-01-01T00:00:00Z&end_date=2024-01-31T23:59:59Z"
```

#### Filtrer par is_anomaly

```bash
# Seulement les anomalies
curl "http://localhost:8084/api/v1/anomalies/?is_anomaly=true"

# Seulement les normaux
curl "http://localhost:8084/api/v1/anomalies/?is_anomaly=false"
```

#### Filtrer par criticité

```bash
curl "http://localhost:8084/api/v1/anomalies/?criticality=high"
```

Valeurs possibles : `low`, `medium`, `high`, `critical`

#### Combinaison de filtres

```bash
curl "http://localhost:8084/api/v1/anomalies/?asset_id=ASSET001&criticality=high&is_anomaly=true&start_date=2024-01-01T00:00:00Z"
```

#### Pagination

```bash
# Première page (100 résultats)
curl "http://localhost:8084/api/v1/anomalies/?limit=100&offset=0"

# Deuxième page
curl "http://localhost:8084/api/v1/anomalies/?limit=100&offset=100"
```

### Format de réponse

```json
{
  "anomalies": [
    {
      "id": 1,
      "asset_id": "ASSET001",
      "sensor_id": "SENSOR001",
      "timestamp": "2024-01-01T12:00:00Z",
      "final_score": 0.75,
      "is_anomaly": true,
      "criticality": "high",
      "scores": [
        {
          "model_name": "isolation_forest",
          "score": 0.8,
          "threshold": 0.5,
          "is_anomaly": true
        },
        {
          "model_name": "one_class_svm",
          "score": 0.7,
          "threshold": 0.5,
          "is_anomaly": true
        },
        {
          "model_name": "lstm_autoencoder",
          "score": 0.75,
          "threshold": 0.5,
          "is_anomaly": true
        }
      ],
      "features": {
        "rms": 10.5,
        "kurtosis": 2.3
      },
      "metadata": {
        "source": "kafka"
      },
      "created_at": "2024-01-01T12:00:00Z"
    }
  ],
  "total": 42,
  "limit": 100,
  "offset": 0,
  "filters": {
    "asset_id": "ASSET001",
    "sensor_id": null,
    "start_date": null,
    "end_date": null,
    "is_anomaly": null,
    "criticality": null
  }
}
```

## 💻 Utilisation directe du service (exemple Python)

```python
from app.database.postgresql import PostgreSQLService
from app.models.anomaly_data import AnomalyDetectionResult

# Initialiser le service
db_service = PostgreSQLService()

# Insérer une anomalie
anomaly_result = AnomalyDetectionResult(...)
anomaly_id = db_service.insert_anomaly(anomaly_result)

# Récupérer les anomalies
anomalies = db_service.get_anomalies(
    asset_id="ASSET001",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31),
    is_anomaly=True,
    criticality="high",
    limit=100,
    offset=0
)

# Compter les anomalies
count = db_service.get_anomaly_count(
    asset_id="ASSET001",
    is_anomaly=True
)

# Fermer le service
db_service.close()
```

## 📊 Requêtes SQL utiles (pour analyse - optionnel)

### Statistiques par asset

```sql
SELECT 
    asset_id,
    COUNT(*) as total_detections,
    SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomalies_count,
    AVG(final_score) as avg_score,
    MAX(final_score) as max_score
FROM anomaly_detections
GROUP BY asset_id
ORDER BY anomalies_count DESC;
```

### Statistiques par criticité

```sql
SELECT 
    criticality,
    COUNT(*) as count,
    AVG(final_score) as avg_score
FROM anomaly_detections
WHERE is_anomaly = true
GROUP BY criticality
ORDER BY count DESC;
```

### Anomalies récentes

```sql
SELECT 
    asset_id,
    sensor_id,
    timestamp,
    final_score,
    criticality
FROM anomaly_detections
WHERE is_anomaly = true
ORDER BY timestamp DESC
LIMIT 10;
```

### Tendances temporelles

```sql
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as detections,
    SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomalies
FROM anomaly_detections
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour;
```

## ⚙️ Performance (information)

- **Pool de connexions** : ThreadedConnectionPool avec minconn=1, maxconn=10
- **Index optimisés** : Requêtes filtrées par asset_id, sensor_id, timestamp sont rapides
- **JSONB** : Colonnes scores, features, metadata utilisent JSONB pour des requêtes efficaces

## 🔧 Dépannage (si problème)

### Erreur de connexion

```
Erreur lors de l'initialisation du pool PostgreSQL: ...
```

**Solution** : Vérifier que PostgreSQL est démarré et que les credentials sont corrects.

### Table non créée

```
Erreur lors de la création des tables: ...
```

**Solution** : Vérifier les permissions de l'utilisateur PostgreSQL. Il doit avoir les droits CREATE TABLE.

### Requêtes lentes

**Solution** : Vérifier que les index sont créés :

```sql
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'anomaly_detections';
```

## 🛠️ Maintenance (pour plus tard - optionnel)

### Nettoyage des anciennes données

```sql
-- Supprimer les anomalies de plus de 90 jours
DELETE FROM anomaly_detections
WHERE timestamp < NOW() - INTERVAL '90 days';
```

### Archivage

```sql
-- Créer une table d'archive
CREATE TABLE anomaly_detections_archive (LIKE anomaly_detections INCLUDING ALL);

-- Archiver les données anciennes
INSERT INTO anomaly_detections_archive
SELECT * FROM anomaly_detections
WHERE timestamp < NOW() - INTERVAL '90 days';

-- Supprimer les données archivées
DELETE FROM anomaly_detections
WHERE timestamp < NOW() - INTERVAL '90 days';
```

### Statistiques

```sql
-- Mettre à jour les statistiques pour optimiser les requêtes
ANALYZE anomaly_detections;
```

