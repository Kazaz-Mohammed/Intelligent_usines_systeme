# Script de Test Complet - Phase 1 Infrastructure
# Ce script teste tous les composants de l'infrastructure

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  TEST COMPLET - PHASE 1 INFRASTRUCTURE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$testResults = @()
$totalTests = 0
$passedTests = 0
$failedTests = 0

function Test-Service {
    param(
        [string]$ServiceName,
        [string]$TestName,
        [scriptblock]$TestScript
    )
    
    $totalTests++
    Write-Host "[TEST] $TestName..." -ForegroundColor Yellow -NoNewline
    
    try {
        $result = & $TestScript
        if ($LASTEXITCODE -eq 0 -or $result -eq $true) {
            Write-Host " ✅ PASS" -ForegroundColor Green
            $script:passedTests++
            $script:testResults += @{
                Test = $TestName
                Service = $ServiceName
                Status = "PASS"
                Message = "Success"
            }
            return $true
        } else {
            Write-Host " ❌ FAIL" -ForegroundColor Red
            $script:failedTests++
            $script:testResults += @{
                Test = $TestName
                Service = $ServiceName
                Status = "FAIL"
                Message = "Exit code: $LASTEXITCODE"
            }
            return $false
        }
    } catch {
        Write-Host " ❌ FAIL" -ForegroundColor Red
        Write-Host "   Error: $_" -ForegroundColor Red
        $script:failedTests++
        $script:testResults += @{
            Test = $TestName
            Service = $ServiceName
            Status = "FAIL"
            Message = $_.Exception.Message
        }
        return $false
    }
}

# ========================================
# 1. Vérification des Services Docker
# ========================================
Write-Host "=== 1. SERVICES DOCKER ===" -ForegroundColor Cyan
Write-Host ""

$services = @("zookeeper", "kafka", "postgresql", "influxdb", "minio", "redis")

foreach ($service in $services) {
    Test-Service -ServiceName $service -TestName "Service $service est démarré" -TestScript {
        $status = docker ps --filter "name=$service" --format "{{.Status}}"
        if ($status -match "Up") {
            return $true
        }
        return $false
    }
}

Write-Host ""

# ========================================
# 2. Tests PostgreSQL + TimescaleDB
# ========================================
Write-Host "=== 2. POSTGRESQL + TIMESCALEDB ===" -ForegroundColor Cyan
Write-Host ""

Test-Service -ServiceName "postgresql" -TestName "Connexion PostgreSQL" -TestScript {
    docker exec -it postgresql psql -U pmuser -d predictive_maintenance -c "SELECT 1;" 2>&1 | Out-Null
    return $LASTEXITCODE -eq 0
}

Test-Service -ServiceName "postgresql" -TestName "Extension TimescaleDB installée" -TestScript {
    $result = docker exec -it postgresql psql -U pmuser -d predictive_maintenance -t -c "SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb';" 2>&1
    return ($result -match "1")
}

Test-Service -ServiceName "postgresql" -TestName "Table raw_sensor_data existe" -TestScript {
    $result = docker exec -it postgresql psql -U pmuser -d predictive_maintenance -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'raw_sensor_data';" 2>&1
    return ($result -match "1")
}

Test-Service -ServiceName "postgresql" -TestName "Table processed_windows existe" -TestScript {
    $result = docker exec -it postgresql psql -U pmuser -d predictive_maintenance -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'processed_windows';" 2>&1
    return ($result -match "1")
}

Test-Service -ServiceName "postgresql" -TestName "Table assets existe avec données" -TestScript {
    $result = docker exec -it postgresql psql -U pmuser -d predictive_maintenance -t -c "SELECT COUNT(*) FROM assets;" 2>&1
    $count = ($result | Where-Object { $_ -match '^\s*\d+\s*$' }) -replace '\s',''
    if ($count -and [int]$count -ge 1) {
        return $true
    }
    return $false
}

Test-Service -ServiceName "postgresql" -TestName "Hypertable raw_sensor_data configurée" -TestScript {
    $result = docker exec -it postgresql psql -U pmuser -d predictive_maintenance -t -c "SELECT COUNT(*) FROM timescaledb_information.hypertables WHERE hypertable_name = 'raw_sensor_data';" 2>&1
    return ($result -match "1")
}

Test-Service -ServiceName "postgresql" -TestName "Vue v_asset_status existe" -TestScript {
    $result = docker exec -it postgresql psql -U pmuser -d predictive_maintenance -t -c "SELECT COUNT(*) FROM information_schema.views WHERE table_name = 'v_asset_status';" 2>&1
    return ($result -match "1")
}

Test-Service -ServiceName "postgresql" -TestName "Insertion test dans raw_sensor_data" -TestScript {
    docker exec -it postgresql psql -U pmuser -d predictive_maintenance -c "INSERT INTO raw_sensor_data (time, asset_id, sensor_id, value, unit) VALUES (NOW(), 'TEST_ASSET', 'TEST_SENSOR', 99.9, 'test') ON CONFLICT DO NOTHING;" 2>&1 | Out-Null
    return $LASTEXITCODE -eq 0
}

Write-Host ""

# ========================================
# 3. Tests Kafka
# ========================================
Write-Host "=== 3. KAFKA ===" -ForegroundColor Cyan
Write-Host ""

Test-Service -ServiceName "kafka" -TestName "Port Kafka 9092 accessible" -TestScript {
    $result = Test-NetConnection localhost -Port 9092 -WarningAction SilentlyContinue
    return $result.TcpTestSucceeded
}

Test-Service -ServiceName "kafka" -TestName "Topic sensor-data existe" -TestScript {
    Start-Sleep -Seconds 3
    $result = docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092 2>&1
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    $lines = $result -split "`n" | Where-Object { $_ -match "sensor-data" }
    return ($lines.Count -gt 0)
}

Test-Service -ServiceName "kafka" -TestName "Topic preprocessed-data existe" -TestScript {
    Start-Sleep -Seconds 1
    $result = docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092 2>&1
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    $lines = $result -split "`n" | Where-Object { $_ -match "preprocessed-data" }
    return ($lines.Count -gt 0)
}

Test-Service -ServiceName "kafka" -TestName "Topic features existe" -TestScript {
    Start-Sleep -Seconds 1
    $result = docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092 2>&1
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    $lines = $result -split "`n" | Where-Object { $_ -match "features" }
    return ($lines.Count -gt 0)
}

Test-Service -ServiceName "kafka" -TestName "Topic anomalies existe" -TestScript {
    Start-Sleep -Seconds 1
    $result = docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092 2>&1
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    $lines = $result -split "`n" | Where-Object { $_ -match "anomalies" }
    return ($lines.Count -gt 0)
}

Test-Service -ServiceName "kafka" -TestName "Topic rul-predictions existe" -TestScript {
    Start-Sleep -Seconds 1
    $result = docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092 2>&1
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    $lines = $result -split "`n" | Where-Object { $_ -match "rul-predictions" }
    return ($lines.Count -gt 0)
}

Test-Service -ServiceName "kafka" -TestName "Topic maintenance-orders existe" -TestScript {
    Start-Sleep -Seconds 1
    $result = docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092 2>&1
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    $lines = $result -split "`n" | Where-Object { $_ -match "maintenance-orders" }
    return ($lines.Count -gt 0)
}

Write-Host ""

# ========================================
# 4. Tests MinIO
# ========================================
Write-Host "=== 4. MINIO ===" -ForegroundColor Cyan
Write-Host ""

Test-Service -ServiceName "minio" -TestName "Port MinIO 9000 accessible" -TestScript {
    $result = Test-NetConnection localhost -Port 9000 -WarningAction SilentlyContinue
    return $result.TcpTestSucceeded
}

Test-Service -ServiceName "minio" -TestName "Bucket raw-sensor-data existe" -TestScript {
    $result = docker exec -it minio mc ls local 2>&1
    return ($result -match "raw-sensor-data")
}

Test-Service -ServiceName "minio" -TestName "Bucket processed-data existe" -TestScript {
    $result = docker exec -it minio mc ls local 2>&1
    return ($result -match "processed-data")
}

Test-Service -ServiceName "minio" -TestName "Bucket model-artifacts existe" -TestScript {
    $result = docker exec -it minio mc ls local 2>&1
    return ($result -match "model-artifacts")
}

Write-Host ""

# ========================================
# 5. Tests Redis
# ========================================
Write-Host "=== 5. REDIS ===" -ForegroundColor Cyan
Write-Host ""

Test-Service -ServiceName "redis" -TestName "Connexion Redis" -TestScript {
    $result = docker exec -it redis redis-cli -a redispassword PING 2>&1
    return ($result -match "PONG")
}

Test-Service -ServiceName "redis" -TestName "SET/GET Redis" -TestScript {
    docker exec -it redis redis-cli -a redispassword SET "test:phase1" "OK" 2>&1 | Out-Null
    $result = docker exec -it redis redis-cli -a redispassword GET "test:phase1" 2>&1
    docker exec -it redis redis-cli -a redispassword DEL "test:phase1" 2>&1 | Out-Null
    return ($result -match "OK")
}

Write-Host ""

# ========================================
# 6. Tests InfluxDB
# ========================================
Write-Host "=== 6. INFLUXDB ===" -ForegroundColor Cyan
Write-Host ""

Test-Service -ServiceName "influxdb" -TestName "Port InfluxDB 8086 accessible" -TestScript {
    $result = Test-NetConnection localhost -Port 8086 -WarningAction SilentlyContinue
    return $result.TcpTestSucceeded
}

Test-Service -ServiceName "influxdb" -TestName "InfluxDB répond" -TestScript {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8086/health" -TimeoutSec 5 -UseBasicParsing -ErrorAction SilentlyContinue
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

Write-Host ""

# ========================================
# 7. Tests de Connectivité Réseau
# ========================================
Write-Host "=== 7. RÉSEAU DOCKER ===" -ForegroundColor Cyan
Write-Host ""

Test-Service -ServiceName "network" -TestName "Réseau predictive-maintenance-network existe" -TestScript {
    $result = docker network ls --filter "name=predictive-maintenance-network" --format "{{.Name}}"
    return ($result -match "predictive-maintenance-network")
}

Write-Host ""

# ========================================
# RÉSUMÉ DES TESTS
# ========================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RÉSUMÉ DES TESTS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Total de tests : $totalTests" -ForegroundColor White
Write-Host "Tests réussis  : $passedTests" -ForegroundColor Green
Write-Host "Tests échoués  : $failedTests" -ForegroundColor $(if ($failedTests -eq 0) { "Green" } else { "Red" })
Write-Host ""

$successRate = if ($totalTests -gt 0) { [math]::Round(($passedTests / $totalTests) * 100, 2) } else { 0 }
Write-Host "Taux de réussite : $successRate%" -ForegroundColor $(if ($successRate -eq 100) { "Green" } elseif ($successRate -ge 90) { "Yellow" } else { "Red" })
Write-Host ""

if ($failedTests -eq 0) {
    Write-Host "✅ TOUS LES TESTS ONT RÉUSSI !" -ForegroundColor Green
    Write-Host "L'infrastructure Phase 1 est opérationnelle." -ForegroundColor Green
} else {
    Write-Host "⚠️  CERTAINS TESTS ONT ÉCHOUÉ" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Tests échoués :" -ForegroundColor Yellow
    foreach ($test in $testResults) {
        if ($test.Status -eq "FAIL") {
            Write-Host "  - $($test.Service): $($test.Test)" -ForegroundColor Red
            Write-Host "    Message: $($test.Message)" -ForegroundColor Gray
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

# Retourner le code de sortie
if ($failedTests -eq 0) {
    exit 0
} else {
    exit 1
}

