#!/bin/bash

# Script d'initialisation des buckets MinIO
# Ce script doit être exécuté après le démarrage de MinIO

MINIO_CONTAINER="minio"
MINIO_ENDPOINT="http://localhost:9000"
MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}"

# Liste des buckets à créer
BUCKETS=(
    "raw-sensor-data"      # Données brutes des capteurs
    "processed-data"       # Données prétraitées
    "model-artifacts"      # Artefacts de modèles ML
    "mlflow-artifacts"     # Artefacts MLflow
    "backups"              # Backups
)

echo "Initialisation des buckets MinIO..."

# Attendre que MinIO soit prêt
echo "Attente du démarrage de MinIO..."
sleep 10

# Fonction pour créer un bucket
create_bucket() {
    local bucket_name=$1
    
    echo "Création du bucket: $bucket_name"
    
    # Utiliser mc (MinIO Client) pour créer le bucket
    docker exec -it $MINIO_CONTAINER mc alias set local $MINIO_ENDPOINT $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD 2>/dev/null
    
    docker exec -it $MINIO_CONTAINER mc mb "local/$bucket_name" --ignore-existing 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✅ Bucket $bucket_name créé avec succès"
        
        # Configurer la politique publique si nécessaire (optionnel)
        # docker exec -it $MINIO_CONTAINER mc anonymous set download "local/$bucket_name"
    else
        echo "⚠️  Bucket $bucket_name existe déjà ou erreur lors de la création"
    fi
}

# Installer mc (MinIO Client) dans le conteneur si nécessaire
docker exec -it $MINIO_CONTAINER sh -c "command -v mc || apk add --no-cache minio-client" 2>/dev/null

# Créer tous les buckets
for bucket_name in "${BUCKETS[@]}"; do
    create_bucket "$bucket_name"
done

# Lister tous les buckets
echo ""
echo "Buckets MinIO disponibles:"
docker exec -it $MINIO_CONTAINER mc ls local

echo ""
echo "✅ Initialisation des buckets MinIO terminée"

