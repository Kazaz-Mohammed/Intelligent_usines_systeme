# Script pour télécharger les images Docker une par une
# Utile en cas de problèmes de connexion

Write-Host "🔄 Téléchargement des images Docker une par une..." -ForegroundColor Cyan
Write-Host ""

$images = @(
    "timescale/timescaledb:latest-pg16",
    "confluentinc/cp-zookeeper:7.5.0",
    "confluentinc/cp-kafka:7.5.0",
    "influxdb:2.7",
    "redis:7-alpine",
    "minio/minio:latest"
)

foreach ($image in $images) {
    Write-Host "📥 Téléchargement de $image..." -ForegroundColor Yellow
    $attempt = 1
    $maxAttempts = 3
    
    while ($attempt -le $maxAttempts) {
        try {
            docker pull $image
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ $image téléchargé avec succès" -ForegroundColor Green
                break
            }
        } catch {
            Write-Host "⚠️  Tentative $attempt/$maxAttempts échouée pour $image" -ForegroundColor Yellow
        }
        
        if ($attempt -lt $maxAttempts) {
            Write-Host "⏳ Attente de 10 secondes avant nouvelle tentative..." -ForegroundColor Yellow
            Start-Sleep -Seconds 10
        }
        $attempt++
    }
    
    if ($attempt -gt $maxAttempts) {
        Write-Host "❌ Échec du téléchargement de $image après $maxAttempts tentatives" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "✅ Téléchargement terminé" -ForegroundColor Green
Write-Host ""
Write-Host "Vérification des images téléchargées:" -ForegroundColor Cyan
docker images

