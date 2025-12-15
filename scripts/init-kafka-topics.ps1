# Script PowerShell d'initialisation des topics Kafka
# Pour Windows

$KAFKA_CONTAINER = "kafka"
$KAFKA_BOOTSTRAP_SERVER = "localhost:9092"

# Liste des topics à créer
$TOPICS = @(
    @{Name="sensor-data"; Partitions=3; Replicas=1},
    @{Name="preprocessed-data"; Partitions=3; Replicas=1},
    @{Name="features"; Partitions=3; Replicas=1},
    @{Name="anomalies"; Partitions=3; Replicas=1},
    @{Name="rul-predictions"; Partitions=3; Replicas=1},
    @{Name="maintenance-orders"; Partitions=3; Replicas=1}
)

Write-Host "Initialisation des topics Kafka..." -ForegroundColor Cyan

# Attendre que Kafka soit prêt
Write-Host "Attente du démarrage de Kafka..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Fonction pour créer un topic
function Create-Topic {
    param(
        [string]$TopicName,
        [int]$Partitions,
        [int]$Replicas
    )
    
    Write-Host "Création du topic: $TopicName (partitions: $Partitions, replicas: $Replicas)" -ForegroundColor Yellow
    
    $result = docker exec -it $KAFKA_CONTAINER kafka-topics `
        --create `
        --bootstrap-server $KAFKA_BOOTSTRAP_SERVER `
        --topic $TopicName `
        --partitions $Partitions `
        --replication-factor $Replicas `
        --if-not-exists 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Topic $TopicName créé avec succès" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Topic $TopicName existe déjà ou erreur lors de la création" -ForegroundColor Yellow
    }
}

# Créer tous les topics
foreach ($topic in $TOPICS) {
    Create-Topic -TopicName $topic.Name -Partitions $topic.Partitions -Replicas $topic.Replicas
}

# Lister tous les topics
Write-Host "`nTopics Kafka disponibles:" -ForegroundColor Cyan
docker exec -it $KAFKA_CONTAINER kafka-topics --list --bootstrap-server $KAFKA_BOOTSTRAP_SERVER

Write-Host "`n✅ Initialisation des topics Kafka terminée" -ForegroundColor Green

