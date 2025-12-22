package com.predictivemaintenance.ingestion.service;

import com.predictivemaintenance.ingestion.model.SensorData;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Service principal d'ingestion qui orchestre la collecte, normalisation et publication
 */
@Slf4j
@Service
public class IngestionService {

    private final OPCUAService opcuaService;
    private final DataNormalizationService normalizationService;
    private final KafkaProducerService kafkaProducerService;
    private final TimescaleDBService timescaleDBService;
    private final MinIOService minioService;
    
    // Optional: InfluxDB service (only available when influxdb.enabled=true)
    private final InfluxDBService influxDBService;

    @Autowired
    public IngestionService(
            OPCUAService opcuaService,
            DataNormalizationService normalizationService,
            KafkaProducerService kafkaProducerService,
            TimescaleDBService timescaleDBService,
            MinIOService minioService,
            @Autowired(required = false) InfluxDBService influxDBService) {
        this.opcuaService = opcuaService;
        this.normalizationService = normalizationService;
        this.kafkaProducerService = kafkaProducerService;
        this.timescaleDBService = timescaleDBService;
        this.minioService = minioService;
        this.influxDBService = influxDBService;
        
        if (influxDBService != null) {
            log.info("InfluxDB integration enabled");
        } else {
            log.info("InfluxDB integration disabled");
        }
    }

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

            // 5. Archiver dans MinIO (non-blocking, errors are logged but don't stop ingestion)
            try {
                minioService.storeBatch(normalizedDataList);
            } catch (Exception e) {
                log.warn("Failed to archive data in MinIO (continuing anyway): {}", e.getMessage());
            }

            // 6. Stocker dans InfluxDB (non-blocking, optional)
            if (influxDBService != null) {
                try {
                    influxDBService.writeBatch(normalizedDataList);
                } catch (Exception e) {
                    log.warn("Failed to write data to InfluxDB (continuing anyway): {}", e.getMessage());
                }
            }

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
            
            // MinIO storage is optional (non-blocking)
            try {
                minioService.storeSensorData(normalized);
            } catch (Exception e) {
                log.warn("Failed to archive data in MinIO (continuing anyway): {}", e.getMessage());
            }

            // InfluxDB storage is optional (non-blocking)
            if (influxDBService != null) {
                try {
                    influxDBService.writeSensorData(normalized);
                } catch (Exception e) {
                    log.warn("Failed to write data to InfluxDB (continuing anyway): {}", e.getMessage());
                }
            }

            log.debug("Processed sensor data: asset={}, sensor={}", 
                normalized.getAssetId(), normalized.getSensorId());
        } catch (Exception e) {
            log.error("Error processing sensor data", e);
        }
    }
}

