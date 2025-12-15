# Script pour tester la connexion Docker

Write-Host "=== Test de Connexion Docker ===" -ForegroundColor Cyan
Write-Host ""

# Test 1 : Vérifier Docker
Write-Host "[TEST 1] Vérification de Docker..." -ForegroundColor Yellow
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "[OK] Docker est installé" -ForegroundColor Green
    docker --version
} else {
    Write-Host "[ERREUR] Docker n'est pas installé" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Test 2 : Vérifier Docker Desktop
Write-Host "[TEST 2] Vérification de Docker Desktop..." -ForegroundColor Yellow
try {
    $result = docker ps 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Docker Desktop est démarré" -ForegroundColor Green
    } else {
        Write-Host "[ERREUR] Docker Desktop n'est pas démarré" -ForegroundColor Red
        Write-Host "Démarrez Docker Desktop et réessayez" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "[ERREUR] Docker Desktop n'est pas accessible" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Test 3 : Test de connexion à Docker Hub
Write-Host "[TEST 3] Test de connexion à Docker Hub..." -ForegroundColor Yellow
try {
    Write-Host "Téléchargement de l'image hello-world..." -ForegroundColor Cyan
    $result = docker pull hello-world 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Connexion à Docker Hub fonctionne" -ForegroundColor Green
    } else {
        Write-Host "[ERREUR] Connexion à Docker Hub échouée" -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
        Write-Host ""
        Write-Host "Solutions possibles :" -ForegroundColor Yellow
        Write-Host "1. Vérifier la configuration proxy dans Docker Desktop" -ForegroundColor Yellow
        Write-Host "2. Vérifier la configuration DNS dans Docker Desktop" -ForegroundColor Yellow
        Write-Host "3. Vérifier votre connexion Internet" -ForegroundColor Yellow
        Write-Host "4. Utiliser un VPN si vous êtes derrière un pare-feu" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "[ERREUR] Impossible de tester la connexion à Docker Hub" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

Write-Host ""

# Test 4 : Test de téléchargement de l'image Python
Write-Host "[TEST 4] Test de téléchargement de l'image Python..." -ForegroundColor Yellow
try {
    Write-Host "Téléchargement de l'image python:3.11-slim..." -ForegroundColor Cyan
    $result = docker pull python:3.11-slim 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Image Python téléchargée avec succès" -ForegroundColor Green
    } else {
        Write-Host "[ERREUR] Téléchargement de l'image Python échoué" -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
        Write-Host ""
        Write-Host "Solutions possibles :" -ForegroundColor Yellow
        Write-Host "1. Vérifier la configuration proxy dans Docker Desktop" -ForegroundColor Yellow
        Write-Host "2. Essayer sans proxy : docker pull --platform linux/amd64 python:3.11-slim" -ForegroundColor Yellow
        Write-Host "3. Utiliser une image alternative (voir TROUBLESHOOTING_DOCKER.md)" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "[ERREUR] Impossible de télécharger l'image Python" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

Write-Host ""

# Test 5 : Vérifier l'image téléchargée
Write-Host "[TEST 5] Vérification de l'image Python..." -ForegroundColor Yellow
try {
    $result = docker images python:3.11-slim
    if ($result -match "python.*3.11-slim") {
        Write-Host "[OK] Image Python disponible" -ForegroundColor Green
        docker images python:3.11-slim
    } else {
        Write-Host "[ERREUR] Image Python non trouvée" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "[ERREUR] Impossible de vérifier l'image Python" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Tous les tests sont passés ===" -ForegroundColor Green
Write-Host "Vous pouvez maintenant exécuter les tests Docker :" -ForegroundColor Cyan
Write-Host "  .\scripts\run-tests-docker.ps1" -ForegroundColor Yellow

