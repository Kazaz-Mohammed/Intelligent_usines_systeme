package com.predictivemaintenance.ingestion.service;

import com.predictivemaintenance.ingestion.model.SensorData;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Service principal d'ingestion qui orchestre la collecte, normalisation et publication
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class IngestionService {

    private final OPCUAService opcuaService;
    private final DataNormalizationService normalizationService;
    private final KafkaProducerService kafkaProducerService;
    private final TimescaleDBService timescaleDBService;
    private final MinIOService minioService;

    /**
     * Collecte et traite les données depuis toutes les sources
     */
    @Scheduled(fixedRateString = "${app.ingestion.flush-interval:5000}")
    public void collectAndProcessData() {
        try {
            log.debug("Starting data collection cycle");

            // 1. Collecter depuis OPC UA
            List<SensorData> rawDataList = opcuaService.readAllNodes();

            if (rawDataList.isEmpty()) {
                log.debug("No data collected");
                return;
            }

            // 2. Normaliser les données
            List<SensorData> normalizedDataList = rawDataList.stream()
                    .map(normalizationService::normalize)
                    .filter(data -> data != null)
                    .toList();

            // 3. Publier sur Kafka
            kafkaProducerService.publishBatch(normalizedDataList);

            // 4. Stocker dans TimescaleDB
            timescaleDBService.insertBatch(normalizedDataList);

            // 5. Archiver dans MinIO
            minioService.storeBatch(normalizedDataList);

            log.info("Processed {} sensor data records", normalizedDataList.size());

        } catch (Exception e) {
            log.error("Error in data collection cycle", e);
        }
    }

    /**
     * Traite une donnée individuelle
     */
    public void processSensorData(SensorData rawData) {
        try {
            SensorData normalized = normalizationService.normalize(rawData);
            
            kafkaProducerService.publishSensorData(normalized);
            timescaleDBService.insertSensorData(normalized);
            minioService.storeSensorData(normalized);

            log.debug("Processed sensor data: asset={}, sensor={}", 
                normalized.getAssetId(), normalized.getSensorId());
        } catch (Exception e) {
            log.error("Error processing sensor data", e);
        }
    }
}

