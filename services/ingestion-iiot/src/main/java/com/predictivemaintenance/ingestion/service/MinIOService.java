package com.predictivemaintenance.ingestion.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictivemaintenance.ingestion.model.SensorData;
import io.minio.MinioClient;
import io.minio.PutObjectArgs;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.time.format.DateTimeFormatter;
import java.util.List;

/**
 * Service pour stocker les données brutes dans MinIO
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class MinIOService {

    private final MinioClient minioClient;
    private final ObjectMapper objectMapper;

    @Value("${minio.bucket-name:raw-sensor-data}")
    private String bucketName;

    /**
     * Stocke une donnée capteur dans MinIO
     */
    public void storeSensorData(SensorData sensorData) {
        try {
            String objectName = generateObjectName(sensorData);
            String jsonData = objectMapper.writeValueAsString(sensorData);
            
            minioClient.putObject(
                PutObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .contentType("application/json")
                    .stream(new ByteArrayInputStream(jsonData.getBytes(StandardCharsets.UTF_8)), 
                           jsonData.length(), -1)
                    .build()
            );

            log.debug("Stored sensor data in MinIO: {}", objectName);
        } catch (Exception e) {
            log.error("Error storing sensor data in MinIO", e);
            throw new RuntimeException("Failed to store in MinIO", e);
        }
    }

    /**
     * Stocke un batch de données
     */
    public void storeBatch(List<SensorData> sensorDataList) {
        sensorDataList.forEach(this::storeSensorData);
        log.info("Stored batch of {} records in MinIO", sensorDataList.size());
    }

    private String generateObjectName(SensorData sensorData) {
        // Format: year/month/day/asset_id/sensor_id/timestamp.json
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy/MM/dd");
        String datePath = sensorData.getTimestamp().atZone(java.time.ZoneId.of("UTC"))
            .format(formatter);
        
        return String.format("%s/%s/%s/%s_%s.json",
            datePath,
            sensorData.getAssetId(),
            sensorData.getSensorId(),
            sensorData.getSensorId(),
            sensorData.getTimestamp().toEpochMilli()
        );
    }
}

