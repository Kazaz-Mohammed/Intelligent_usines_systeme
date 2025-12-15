# Script pour corriger les problèmes de proxy Docker

Write-Host "=== Correction des Problèmes de Proxy Docker ===" -ForegroundColor Cyan
Write-Host ""

# Instructions pour corriger les problèmes de proxy
Write-Host "Instructions pour corriger les problèmes de proxy Docker :" -ForegroundColor Yellow
Write-Host ""

Write-Host "1. Ouvrir Docker Desktop" -ForegroundColor Green
Write-Host "2. Aller dans Settings > Resources > Proxies" -ForegroundColor Green
Write-Host "3. Vérifier les paramètres de proxy :" -ForegroundColor Green
Write-Host "   - Si vous utilisez un proxy, configurez-le correctement" -ForegroundColor Cyan
Write-Host "   - Si vous n'utilisez pas de proxy, désactivez-le" -ForegroundColor Cyan
Write-Host "4. Redémarrer Docker Desktop" -ForegroundColor Green
Write-Host ""

# Vérifier la configuration actuelle
Write-Host "Vérification de la configuration Docker actuelle..." -ForegroundColor Yellow
Write-Host ""

try {
    $dockerInfo = docker info 2>&1
    if ($dockerInfo -match "Proxy") {
        Write-Host "Configuration proxy détectée :" -ForegroundColor Yellow
        $dockerInfo | Select-String -Pattern "Proxy" | ForEach-Object { Write-Host $_ -ForegroundColor Cyan }
    } else {
        Write-Host "[INFO] Aucune configuration proxy détectée" -ForegroundColor Green
    }
} catch {
    Write-Host "[ERREUR] Impossible de vérifier la configuration Docker" -ForegroundColor Red
}

Write-Host ""

# Test de connexion
Write-Host "Test de connexion à Docker Hub..." -ForegroundColor Yellow
Write-Host ""

try {
    Write-Host "Téléchargement de l'image hello-world..." -ForegroundColor Cyan
    $result = docker pull hello-world 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Connexion à Docker Hub fonctionne" -ForegroundColor Green
    } else {
        Write-Host "[ERREUR] Connexion à Docker Hub échouée" -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
        Write-Host ""
        Write-Host "Solutions supplémentaires :" -ForegroundColor Yellow
        Write-Host "1. Vérifier votre connexion Internet" -ForegroundColor Cyan
        Write-Host "2. Vérifier les paramètres DNS dans Docker Desktop" -ForegroundColor Cyan
        Write-Host "3. Utiliser un VPN si vous êtes derrière un pare-feu" -ForegroundColor Cyan
        Write-Host "4. Contacter l'administrateur réseau si vous êtes dans une entreprise" -ForegroundColor Cyan
    }
} catch {
    Write-Host "[ERREUR] Impossible de tester la connexion" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Instructions complètes ===" -ForegroundColor Cyan
Write-Host "Consultez TROUBLESHOOTING_DOCKER.md pour plus de détails" -ForegroundColor Yellow

