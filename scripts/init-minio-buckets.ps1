# Script PowerShell d'initialisation des buckets MinIO
# Pour Windows

$MINIO_CONTAINER = "minio"
$MINIO_ENDPOINT = "http://localhost:9000"
$MINIO_ROOT_USER = if ($env:MINIO_ROOT_USER) { $env:MINIO_ROOT_USER } else { "minioadmin" }
$MINIO_ROOT_PASSWORD = if ($env:MINIO_ROOT_PASSWORD) { $env:MINIO_ROOT_PASSWORD } else { "minioadmin" }

# Liste des buckets à créer
$BUCKETS = @(
    "raw-sensor-data",
    "processed-data",
    "model-artifacts",
    "mlflow-artifacts",
    "backups"
)

Write-Host "Initialisation des buckets MinIO..." -ForegroundColor Cyan

# Attendre que MinIO soit prêt
Write-Host "Attente du démarrage de MinIO..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Installer mc (MinIO Client) dans le conteneur si nécessaire
Write-Host "Vérification de MinIO Client..." -ForegroundColor Yellow
docker exec -it $MINIO_CONTAINER sh -c "command -v mc || (apk add --no-cache minio-client 2>/dev/null || echo 'mc not available')" 2>&1 | Out-Null

# Fonction pour créer un bucket
function Create-Bucket {
    param(
        [string]$BucketName
    )
    
    Write-Host "Création du bucket: $BucketName" -ForegroundColor Yellow
    
    # Configurer l'alias MinIO
    docker exec -it $MINIO_CONTAINER mc alias set local $MINIO_ENDPOINT $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD 2>&1 | Out-Null
    
    # Créer le bucket
    $result = docker exec -it $MINIO_CONTAINER mc mb "local/$BucketName" --ignore-existing 2>&1
    
    if ($LASTEXITCODE -eq 0 -or $result -match "already exists") {
        Write-Host "✅ Bucket $BucketName créé/existe déjà" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Erreur lors de la création du bucket $BucketName" -ForegroundColor Yellow
        Write-Host $result -ForegroundColor Red
    }
}

# Créer tous les buckets
foreach ($bucket in $BUCKETS) {
    Create-Bucket -BucketName $bucket
}

# Lister tous les buckets
Write-Host "`nBuckets MinIO disponibles:" -ForegroundColor Cyan
docker exec -it $MINIO_CONTAINER mc ls local 2>&1

Write-Host "`n✅ Initialisation des buckets MinIO terminée" -ForegroundColor Green

