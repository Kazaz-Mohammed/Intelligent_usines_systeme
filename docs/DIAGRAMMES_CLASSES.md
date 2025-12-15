# Diagrammes de Classes - Microservices

## Vue d'Ensemble

Ce document présente les diagrammes de classes pour chaque microservice de la plateforme de maintenance prédictive.

---

## 1. Service : Ingestion-IIoT

### Diagramme de Classes

```mermaid
classDiagram
    class IngestionIiotApplication {
        +main(String[] args)
    }
    
    class IngestionController {
        -IngestionService ingestionService
        +getHealth() ResponseEntity
        +startIngestion() ResponseEntity
        +stopIngestion() ResponseEntity
    }
    
    class IngestionService {
        -OPCUAService opcuaService
        -KafkaProducerService kafkaProducer
        -TimescaleDBService timescaleDB
        -MinIOService minioService
        -DataNormalizationService normalizationService
        +startIngestion()
        +stopIngestion()
        +processSensorData(SensorData)
    }
    
    class OPCUAService {
        -OPCUAConfig config
        -UaClient client
        +connect()
        +disconnect()
        +readNode(String nodeId) CompletableFuture
        +subscribeToNodes()
    }
    
    class KafkaProducerService {
        -KafkaTemplate kafkaTemplate
        -String topicName
        +publishSensorData(SensorData)
        +publishBatch(List~SensorData~)
    }
    
    class TimescaleDBService {
        -JdbcTemplate jdbcTemplate
        +saveSensorData(SensorData)
        +batchSave(List~SensorData~)
        +queryHistoricalData(String assetId, LocalDateTime start, LocalDateTime end)
    }
    
    class MinIOService {
        -MinioClient minioClient
        -String bucketName
        +uploadData(String key, byte[] data)
        +downloadData(String key) byte[]
        +archiveSensorData(SensorData)
    }
    
    class DataNormalizationService {
        +normalize(SensorData) SensorData
        +validate(SensorData) boolean
        +convertUnit(SensorData, String targetUnit) SensorData
    }
    
    class SensorData {
        -String assetId
        -String sensorId
        -Double value
        -String unit
        -LocalDateTime timestamp
        -Map~String,Object~ metadata
        +getAssetId() String
        +getSensorId() String
        +getValue() Double
        +getTimestamp() LocalDateTime
    }
    
    class OPCUAConfig {
        -String endpointUrl
        -int connectionTimeout
        -List~NodeConfig~ nodes
        +getEndpointUrl() String
        +getNodes() List~NodeConfig~
    }
    
    class NodeConfig {
        -String nodeId
        -String assetId
        -String sensorId
        -String unit
    }
    
    IngestionIiotApplication --> IngestionController
    IngestionController --> IngestionService
    IngestionService --> OPCUAService
    IngestionService --> KafkaProducerService
    IngestionService --> TimescaleDBService
    IngestionService --> MinIOService
    IngestionService --> DataNormalizationService
    IngestionService --> SensorData
    OPCUAService --> OPCUAConfig
    OPCUAConfig --> NodeConfig
    KafkaProducerService --> SensorData
    TimescaleDBService --> SensorData
    MinIOService --> SensorData
    DataNormalizationService --> SensorData
```

### Description des Classes

#### IngestionController
- **Responsabilité** : Point d'entrée REST pour contrôler l'ingestion
- **Méthodes principales** :
  - `getHealth()` : Vérification de l'état du service
  - `startIngestion()` : Démarrage de la collecte
  - `stopIngestion()` : Arrêt de la collecte

#### IngestionService
- **Responsabilité** : Orchestration de la collecte et du traitement des données
- **Méthodes principales** :
  - `startIngestion()` : Initialise la connexion OPC UA et démarre la collecte
  - `processSensorData()` : Traite chaque donnée collectée

#### OPCUAService
- **Responsabilité** : Communication avec les serveurs OPC UA
- **Méthodes principales** :
  - `connect()` : Établit la connexion
  - `readNode()` : Lit une valeur depuis un node OPC UA
  - `subscribeToNodes()` : S'abonne aux changements de valeurs

#### SensorData
- **Responsabilité** : Modèle de données pour les valeurs capteurs
- **Attributs** :
  - `assetId` : Identifiant de l'équipement
  - `sensorId` : Identifiant du capteur
  - `value` : Valeur mesurée
  - `timestamp` : Horodatage

---

## 2. Service : Prétraitement

### Diagramme de Classes

```mermaid
classDiagram
    class PreprocessingApplication {
        +main()
    }
    
    class PreprocessingController {
        -PreprocessingService preprocessingService
        +getHealth() dict
        +processData(SensorData) PreprocessedData
        +getStats() dict
    }
    
    class PreprocessingService {
        -CleaningService cleaningService
        -ResamplingService resamplingService
        -DenoisingService denoisingService
        -KafkaConsumerService kafkaConsumer
        -KafkaProducerService kafkaProducer
        -TimescaleDBService timescaleDB
        +process(SensorData) PreprocessedData
        +processBatch(List~SensorData~) List~PreprocessedData~
    }
    
    class CleaningService {
        +removeOutliers(DataFrame) DataFrame
        +fillMissingValues(DataFrame) DataFrame
        +validateData(DataFrame) bool
    }
    
    class ResamplingService {
        +resample(DataFrame, str frequency) DataFrame
        +interpolate(DataFrame) DataFrame
    }
    
    class DenoisingService {
        +applyFilter(DataFrame, str filterType) DataFrame
        +applyWaveletDenoising(DataFrame) DataFrame
    }
    
    class FrequencyAnalysisService {
        +computeFFT(DataFrame) dict
        +computePowerSpectrum(DataFrame) dict
    }
    
    class WindowingService {
        +createWindows(DataFrame, int windowSize, int overlap) List~WindowedData~
    }
    
    class KafkaConsumerService {
        -Consumer consumer
        +consume() Generator~SensorData~
        +startConsumer()
        +stopConsumer()
    }
    
    class KafkaProducerService {
        -Producer producer
        +publish(PreprocessedData)
        +publishBatch(List~PreprocessedData~)
    }
    
    class SensorData {
        +assetId: str
        +sensorId: str
        +value: float
        +timestamp: datetime
    }
    
    class PreprocessedData {
        +assetId: str
        +sensorId: str
        +values: List~float~
        +timestamps: List~datetime~
        +metadata: dict
    }
    
    class WindowedData {
        +windowId: str
        +data: DataFrame
        +startTime: datetime
        +endTime: datetime
    }
    
    PreprocessingApplication --> PreprocessingController
    PreprocessingController --> PreprocessingService
    PreprocessingService --> CleaningService
    PreprocessingService --> ResamplingService
    PreprocessingService --> DenoisingService
    PreprocessingService --> FrequencyAnalysisService
    PreprocessingService --> WindowingService
    PreprocessingService --> KafkaConsumerService
    PreprocessingService --> KafkaProducerService
    PreprocessingService --> SensorData
    PreprocessingService --> PreprocessedData
    WindowingService --> WindowedData
```

---

## 3. Service : Extraction-Features

### Diagramme de Classes

```mermaid
classDiagram
    class ExtractionFeaturesApplication {
        +main()
    }
    
    class FeaturesController {
        -FeatureExtractionService featureService
        +getHealth() dict
        +extractFeatures(PreprocessedData) FeatureData
        +getFeaturesByAsset(str assetId) List~FeatureData~
    }
    
    class FeatureExtractionService {
        -TemporalFeaturesService temporalService
        -FrequencyFeaturesService frequencyService
        -WaveletFeaturesService waveletService
        -TSFreshFeaturesService tsfreshService
        -StandardizationService standardizationService
        -FeastService feastService
        -KafkaConsumerService kafkaConsumer
        -KafkaProducerService kafkaProducer
        +extract(PreprocessedData) FeatureData
        +extractBatch(List~PreprocessedData~) List~FeatureData~
    }
    
    class TemporalFeaturesService {
        +computeRMS(DataFrame) float
        +computeMean(DataFrame) float
        +computeStd(DataFrame) float
        +computeSkewness(DataFrame) float
        +computeKurtosis(DataFrame) float
        +computeAll(DataFrame) dict
    }
    
    class FrequencyFeaturesService {
        +computeFFTFeatures(DataFrame) dict
        +computeSpectralFeatures(DataFrame) dict
        +computeBandPower(DataFrame, float low, float high) float
    }
    
    class WaveletFeaturesService {
        +computeWaveletCoefficients(DataFrame, str wavelet) dict
        +computeEnergyByLevel(DataFrame) dict
    }
    
    class TSFreshFeaturesService {
        +extractComprehensive(DataFrame) dict
    }
    
    class StandardizationService {
        +standardizeByAssetType(FeatureData, str assetType) FeatureData
        +normalize(FeatureData) FeatureData
    }
    
    class FeastService {
        -FeatureStore featureStore
        +saveFeatures(FeatureData)
        +getFeatures(str entityId) dict
    }
    
    class FeatureData {
        +assetId: str
        +sensorId: str
        +timestamp: datetime
        +temporalFeatures: dict
        +frequencyFeatures: dict
        +waveletFeatures: dict
        +metadata: dict
    }
    
    ExtractionFeaturesApplication --> FeaturesController
    FeaturesController --> FeatureExtractionService
    FeatureExtractionService --> TemporalFeaturesService
    FeatureExtractionService --> FrequencyFeaturesService
    FeatureExtractionService --> WaveletFeaturesService
    FeatureExtractionService --> TSFreshFeaturesService
    FeatureExtractionService --> StandardizationService
    FeatureExtractionService --> FeastService
    FeatureExtractionService --> FeatureData
```

---

## 4. Service : Détection-Anomalies

### Diagramme de Classes

```mermaid
classDiagram
    class DetectionAnomaliesApplication {
        +main()
    }
    
    class AnomaliesController {
        -AnomalyDetectionService detectionService
        +getHealth() dict
        +detectAnomalies(FeatureData) AnomalyResult
        +getAnomaliesByAsset(str assetId) List~AnomalyResult~
        +trainModels(TrainingData) dict
    }
    
    class AnomalyDetectionService {
        -IsolationForestService isolationForest
        -OneClassSVMService oneClassSVM
        -LSTMAutoencoderService lstmAutoencoder
        -MLflowService mlflowService
        +detect(FeatureData) AnomalyResult
        +detectEnsemble(FeatureData) AnomalyResult
        +trainAllModels(TrainingData)
    }
    
    class IsolationForestService {
        -IsolationForest model
        +train(DataFrame)
        +predict(DataFrame) AnomalyResult
        +getModel() IsolationForest
    }
    
    class OneClassSVMService {
        -OneClassSVM model
        +train(DataFrame)
        +predict(DataFrame) AnomalyResult
    }
    
    class LSTMAutoencoderService {
        -torch.nn.Module model
        -device device
        +train(DataFrame, int epochs)
        +predict(DataFrame) AnomalyResult
        +loadModel(str path)
        +saveModel(str path)
    }
    
    class MLflowService {
        -MlflowClient client
        +logModel(str name, object model, dict metrics)
        +loadModel(str name, int version) object
        +getModelVersions(str name) List~dict~
    }
    
    class KafkaConsumerService {
        +consume() Generator~FeatureData~
    }
    
    class KafkaProducerService {
        +publish(AnomalyResult)
    }
    
    class AnomalyResult {
        +assetId: str
        +sensorId: str
        +timestamp: datetime
        +isAnomaly: bool
        +anomalyScore: float
        +criticality: str
        +modelUsed: str
        +features: dict
    }
    
    DetectionAnomaliesApplication --> AnomaliesController
    AnomaliesController --> AnomalyDetectionService
    AnomalyDetectionService --> IsolationForestService
    AnomalyDetectionService --> OneClassSVMService
    AnomalyDetectionService --> LSTMAutoencoderService
    AnomalyDetectionService --> MLflowService
    AnomalyDetectionService --> AnomalyResult
```

---

## 5. Service : Prédiction-RUL

### Diagramme de Classes

```mermaid
classDiagram
    class PredictionRulApplication {
        +main()
    }
    
    class RulController {
        -RULPredictionService predictionService
        +getHealth() dict
        +predictRUL(FeatureData) RULPrediction
        +getPredictionsByAsset(str assetId) List~RULPrediction~
        +trainModels(TrainingData) dict
    }
    
    class RULPredictionService {
        -LSTMService lstmService
        -GRUService gruService
        -TCNService tcnService
        -XGBoostService xgboostService
        -CalibrationService calibrationService
        -MLflowService mlflowService
        +predict(FeatureData) RULPrediction
        +predictEnsemble(FeatureData) RULPrediction
        +trainAllModels(TrainingData)
    }
    
    class LSTMService {
        -torch.nn.Module model
        +train(DataFrame, DataFrame targets)
        +predict(DataFrame) float
    }
    
    class GRUService {
        -torch.nn.Module model
        +train(DataFrame, DataFrame targets)
        +predict(DataFrame) float
    }
    
    class TCNService {
        -torch.nn.Module model
        +train(DataFrame, DataFrame targets)
        +predict(DataFrame) float
    }
    
    class XGBoostService {
        -XGBRegressor model
        +train(DataFrame, DataFrame targets)
        +predict(DataFrame) float
    }
    
    class CalibrationService {
        +calibratePredictions(List~float~ predictions, List~float~ actuals) CalibratedPredictions
        +computeConfidenceInterval(float prediction, float uncertainty) tuple
    }
    
    class TransferLearningService {
        +transferFromPretrained(str sourceModel, str targetAsset) Model
        +fineTune(Model model, DataFrame data)
    }
    
    class RULPrediction {
        +assetId: str
        +sensorId: str
        +timestamp: datetime
        +rulPrediction: float
        +confidenceIntervalLower: float
        +confidenceIntervalUpper: float
        +uncertainty: float
        +modelUsed: str
        +modelScores: dict
    }
    
    PredictionRulApplication --> RulController
    RulController --> RULPredictionService
    RULPredictionService --> LSTMService
    RULPredictionService --> GRUService
    RULPredictionService --> TCNService
    RULPredictionService --> XGBoostService
    RULPredictionService --> CalibrationService
    RULPredictionService --> TransferLearningService
    RULPredictionService --> RULPrediction
```

---

## 6. Service : Orchestrateur-Maintenance

### Diagramme de Classes

```mermaid
classDiagram
    class OrchestrateurMaintenanceApplication {
        +main(String[] args)
    }
    
    class InterventionController {
        -WorkOrderService workOrderService
        +createIntervention(InterventionRequest) WorkOrder
        +getInterventions() List~WorkOrder~
    }
    
    class WorkOrderController {
        -WorkOrderService workOrderService
        +getWorkOrders() List~WorkOrder~
        +getWorkOrder(Long id) WorkOrder
        +updateWorkOrder(Long id, WorkOrder) WorkOrder
    }
    
    class WorkOrderService {
        -WorkOrderRepository repository
        -DroolsRuleService droolsService
        -OptimizationService optimizationService
        -PlanningService planningService
        +createWorkOrder(InterventionRequest) WorkOrder
        +updateWorkOrder(WorkOrder) WorkOrder
        +getWorkOrdersByStatus(WorkOrderStatus) List~WorkOrder~
    }
    
    class DroolsRuleService {
        -KieContainer kieContainer
        +evaluateIntervention(InterventionRequest) MaintenanceDecision
        +applyRules(InterventionRequest) InterventionRequest
    }
    
    class OptimizationService {
        -Solver solver
        +optimizeSchedule(List~WorkOrder~, List~Technician~) MaintenancePlan
        +solveWithORTools(List~WorkOrder~) MaintenancePlan
    }
    
    class PlanningService {
        -OptimizationService optimizationService
        +createPlan(List~WorkOrder~) MaintenancePlan
        +adjustPlan(MaintenancePlan, WorkOrder) MaintenancePlan
    }
    
    class KafkaConsumerService {
        -KafkaOrchestrationService orchestrationService
        +consumeAnomaly(String message)
        +consumeRUL(String message)
    }
    
    class KafkaOrchestrationService {
        -WorkOrderService workOrderService
        +processAnomalyIntervention(InterventionRequest)
        +processRULIntervention(InterventionRequest)
    }
    
    class KafkaProducerService {
        -KafkaTemplate kafkaTemplate
        +publishWorkOrder(WorkOrder)
        +publishMaintenancePlan(MaintenancePlan)
    }
    
    class WorkOrder {
        -Long id
        -String assetId
        -String description
        -PriorityLevel priority
        -WorkOrderStatus status
        -LocalDateTime scheduledDate
        -String technicianId
        -Map~String,Object~ metadata
    }
    
    class InterventionRequest {
        -String assetId
        -String sensorId
        -InterventionType type
        -AnomalyContext anomalyContext
        -RulContext rulContext
        -LocalDateTime timestamp
    }
    
    class MaintenancePlan {
        -Long id
        -List~WorkOrder~ workOrders
        -LocalDateTime startDate
        -LocalDateTime endDate
        -double totalCost
        -Map~String,Object~ optimizationMetrics
    }
    
    class MaintenanceDecision {
        -boolean interventionRequired
        -PriorityLevel priority
        -String reason
        -LocalDateTime recommendedDate
    }
    
    OrchestrateurMaintenanceApplication --> InterventionController
    OrchestrateurMaintenanceApplication --> WorkOrderController
    InterventionController --> WorkOrderService
    WorkOrderController --> WorkOrderService
    WorkOrderService --> DroolsRuleService
    WorkOrderService --> OptimizationService
    WorkOrderService --> PlanningService
    WorkOrderService --> WorkOrder
    DroolsRuleService --> MaintenanceDecision
    OptimizationService --> MaintenancePlan
    PlanningService --> MaintenancePlan
    KafkaConsumerService --> KafkaOrchestrationService
    KafkaOrchestrationService --> WorkOrderService
    WorkOrderService --> InterventionRequest
```

---

## 7. Service : Dashboard-Monitoring

### Diagramme de Classes

```mermaid
classDiagram
    class DashboardMonitoringApplication {
        +main(String[] args)
    }
    
    class MonitoringController {
        -MonitoringService monitoringService
        +getHealth() ResponseEntity
        +getMetrics() MetricsResponse
        +getAlerts() List~Alert~
        +getServiceStatus() Map~String,ServiceStatus~
    }
    
    class MonitoringService {
        -MetricsAggregationService metricsService
        -AlertService alertService
        -ServiceHealthService healthService
        -KafkaConsumerService kafkaConsumer
        +aggregateMetrics() MetricsResponse
        +checkAlerts() List~Alert~
        +getServiceHealth() Map~String,ServiceStatus~
    }
    
    class MetricsAggregationService {
        -MetricsRepository repository
        +aggregateSystemMetrics() SystemMetrics
        +aggregateServiceMetrics() Map~String,ServiceMetrics~
        +getHistoricalMetrics(String service, LocalDateTime start, LocalDateTime end) List~Metric~
    }
    
    class AlertService {
        -AlertRepository repository
        +checkThresholds(MetricsResponse) List~Alert~
        +createAlert(Alert alert)
        +resolveAlert(Long alertId)
    }
    
    class ServiceHealthService {
        -WebClient webClient
        +checkServiceHealth(String serviceUrl) ServiceStatus
        +checkAllServices() Map~String,ServiceStatus~
    }
    
    class KafkaConsumerService {
        +consumeAllTopics() Generator~Event~
    }
    
    class WebSocketHandler {
        -MonitoringService monitoringService
        +handleConnection(WebSocketSession)
        +sendMetrics(WebSocketSession, MetricsResponse)
    }
    
    class MetricsResponse {
        +systemMetrics: SystemMetrics
        +serviceMetrics: Map~String,ServiceMetrics~
        +timestamp: LocalDateTime
    }
    
    class Alert {
        +id: Long
        +service: String
        +severity: AlertSeverity
        +message: String
        +timestamp: LocalDateTime
        +resolved: boolean
    }
    
    class ServiceStatus {
        +service: String
        +status: HealthStatus
        +responseTime: long
        +lastCheck: LocalDateTime
    }
    
    DashboardMonitoringApplication --> MonitoringController
    MonitoringController --> MonitoringService
    MonitoringService --> MetricsAggregationService
    MonitoringService --> AlertService
    MonitoringService --> ServiceHealthService
    MonitoringService --> MetricsResponse
    MetricsAggregationService --> SystemMetrics
    AlertService --> Alert
    ServiceHealthService --> ServiceStatus
```

---

## 8. Relations Inter-Services

### Diagramme de Relations Globales

```mermaid
classDiagram
    class IngestionService {
        +SensorData
    }
    
    class PreprocessingService {
        +PreprocessedData
    }
    
    class FeatureExtractionService {
        +FeatureData
    }
    
    class AnomalyDetectionService {
        +AnomalyResult
    }
    
    class RULPredictionService {
        +RULPrediction
    }
    
    class OrchestrateurService {
        +WorkOrder
        +MaintenancePlan
    }
    
    class MonitoringService {
        +MetricsResponse
        +Alert
    }
    
    IngestionService -->|Kafka| PreprocessingService : sensor-data
    PreprocessingService -->|Kafka| FeatureExtractionService : preprocessed-data
    FeatureExtractionService -->|Kafka| AnomalyDetectionService : extracted-features
    FeatureExtractionService -->|Kafka| RULPredictionService : extracted-features
    AnomalyDetectionService -->|Kafka| OrchestrateurService : anomalies-detected
    RULPredictionService -->|Kafka| OrchestrateurService : rul-predictions
    OrchestrateurService -->|REST| MonitoringService
    MonitoringService -->|WebSocket| DashboardUsine
```

---

## Conclusion

Ces diagrammes de classes illustrent la structure interne de chaque microservice, montrant les responsabilités, les relations et les dépendances entre les composants. Cette modélisation facilite la compréhension, la maintenance et l'évolution du système.

