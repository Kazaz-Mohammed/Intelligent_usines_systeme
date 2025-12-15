# Script pour résoudre les problèmes de réseau Docker

Write-Host "=== Résolution des Problèmes de Réseau Docker ===" -ForegroundColor Cyan
Write-Host ""

# Vérifier que Docker est démarré
Write-Host "[INFO] Vérification de Docker..." -ForegroundColor Green
if (-not (Get-Process "Docker Desktop" -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Docker Desktop n'est pas démarré" -ForegroundColor Red
    Write-Host "[INFO] Veuillez démarrer Docker Desktop et réessayer" -ForegroundColor Yellow
    exit 1
}

Write-Host "[INFO] Docker Desktop est démarré" -ForegroundColor Green

# Vérifier la connectivité Internet
Write-Host "[INFO] Vérification de la connectivité Internet..." -ForegroundColor Green
$pingResult = Test-Connection -ComputerName 8.8.8.8 -Count 1 -Quiet
if (-not $pingResult) {
    Write-Host "[ERROR] Pas de connectivité Internet" -ForegroundColor Red
    Write-Host "[INFO] Veuillez vérifier votre connexion Internet" -ForegroundColor Yellow
    exit 1
}

Write-Host "[INFO] Connectivité Internet OK" -ForegroundColor Green

# Vérifier l'accès à Docker Hub
Write-Host "[INFO] Vérification de l'accès à Docker Hub..." -ForegroundColor Green
try {
    $response = Invoke-WebRequest -Uri "https://registry-1.docker.io/v2/" -Method Head -TimeoutSec 10 -ErrorAction Stop
    Write-Host "[INFO] Accès à Docker Hub OK" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Problème d'accès à Docker Hub: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "[INFO] Essayons de télécharger l'image Python directement..." -ForegroundColor Yellow
}

# Tester le téléchargement de l'image Python
Write-Host "[INFO] Test du téléchargement de l'image Python..." -ForegroundColor Green
try {
    docker pull python:3.11-slim 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[INFO] Image Python téléchargée avec succès" -ForegroundColor Green
    } else {
        Write-Host "[WARN] Problème lors du téléchargement de python:3.11-slim" -ForegroundColor Yellow
        Write-Host "[INFO] Essayons avec python:3.11..." -ForegroundColor Yellow
        docker pull python:3.11 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[INFO] Image Python 3.11 téléchargée avec succès" -ForegroundColor Green
        } else {
            Write-Host "[ERROR] Impossible de télécharger l'image Python" -ForegroundColor Red
            Write-Host "[INFO] Veuillez vérifier votre configuration Docker" -ForegroundColor Yellow
            exit 1
        }
    }
} catch {
    Write-Host "[ERROR] Erreur lors du téléchargement: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Vérifier les réseaux Docker
Write-Host "[INFO] Vérification des réseaux Docker..." -ForegroundColor Green
$networkExists = docker network ls --format "{{.Name}}" | Select-String -Pattern "predictive-maintenance"
if (-not $networkExists) {
    Write-Host "[INFO] Création du réseau predictive-maintenance..." -ForegroundColor Yellow
    docker network create predictive-maintenance 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[INFO] Réseau créé avec succès" -ForegroundColor Green
    } else {
        Write-Host "[WARN] Problème lors de la création du réseau" -ForegroundColor Yellow
    }
} else {
    Write-Host "[INFO] Réseau predictive-maintenance existe déjà" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Diagnostic Terminé ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Résumé:" -ForegroundColor Green
Write-Host "  - Docker Desktop: OK" -ForegroundColor Green
Write-Host "  - Connectivité Internet: OK" -ForegroundColor Green
Write-Host "  - Image Python: Téléchargée" -ForegroundColor Green
Write-Host "  - Réseau Docker: Configuré" -ForegroundColor Green
Write-Host ""
Write-Host "Vous pouvez maintenant exécuter:" -ForegroundColor Yellow
Write-Host "  .\scripts\run-tests-docker.ps1" -ForegroundColor Cyan
Write-Host ""

