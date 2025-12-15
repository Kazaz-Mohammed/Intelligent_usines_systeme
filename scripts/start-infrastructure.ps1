# Script PowerShell de démarrage de l'infrastructure complète
# Pour Windows

Write-Host "🚀 Démarrage de l'infrastructure Predictive Maintenance..." -ForegroundColor Cyan
Write-Host ""

# Vérifier que Docker est en cours d'exécution
try {
    docker info | Out-Null
} catch {
    Write-Host "❌ Erreur: Docker n'est pas en cours d'exécution" -ForegroundColor Red
    exit 1
}

# Aller dans le répertoire du projet
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_DIR = Split-Path -Parent $SCRIPT_DIR

Set-Location "$PROJECT_DIR\infrastructure"

# Copier .env.example vers .env si .env n'existe pas
if (-not (Test-Path ".env")) {
    if (Test-Path "..\.env.example") {
        Write-Host "📋 Copie de .env.example vers .env..." -ForegroundColor Yellow
        Copy-Item "..\.env.example" ".env"
        Write-Host "⚠️  N'oubliez pas de modifier .env avec vos valeurs de production!" -ForegroundColor Yellow
    } else {
        Write-Host "⚠️  .env.example non trouvé, création de .env vide..." -ForegroundColor Yellow
        New-Item -ItemType File -Path ".env" | Out-Null
    }
}

# Démarrer les services
Write-Host "🐳 Démarrage des conteneurs Docker..." -ForegroundColor Cyan
docker-compose up -d

# Attendre que les services soient prêts
Write-Host ""
Write-Host "⏳ Attente du démarrage des services..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Vérifier l'état des services
Write-Host ""
Write-Host "📊 État des services:" -ForegroundColor Cyan
docker-compose ps

# Initialiser les bases de données (via init script dans PostgreSQL)
Write-Host ""
Write-Host "📦 PostgreSQL sera initialisé automatiquement via init script..." -ForegroundColor Green

# Initialiser les topics Kafka
Write-Host ""
Write-Host "📨 Initialisation des topics Kafka..." -ForegroundColor Cyan
& "$PROJECT_DIR\scripts\init-kafka-topics.ps1"

# Initialiser les buckets MinIO
Write-Host ""
Write-Host "🪣 Initialisation des buckets MinIO..." -ForegroundColor Cyan
& "$PROJECT_DIR\scripts\init-minio-buckets.ps1"

Write-Host ""
Write-Host "✅ Infrastructure démarrée avec succès!" -ForegroundColor Green
Write-Host ""
Write-Host "📍 Services disponibles:" -ForegroundColor Cyan
Write-Host "   - Kafka: localhost:9092"
Write-Host "   - Kafka UI: http://localhost:8080 (si activé avec --profile tools)"
Write-Host "   - PostgreSQL: localhost:5432"
Write-Host "   - pgAdmin: http://localhost:5050 (si activé avec --profile tools)"
Write-Host "   - InfluxDB: http://localhost:8086"
Write-Host "   - MinIO Console: http://localhost:9001"
Write-Host "   - Redis: localhost:6379"
Write-Host ""
Write-Host "📝 Pour arrêter l'infrastructure: docker-compose down" -ForegroundColor Yellow
Write-Host "📝 Pour voir les logs: docker-compose logs -f" -ForegroundColor Yellow

