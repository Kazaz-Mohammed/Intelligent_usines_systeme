package com.predictivemaintenance.ingestion.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictivemaintenance.ingestion.model.SensorData;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.sql.Timestamp;
import java.util.List;
import java.util.Map;

/**
 * Service pour stocker les données dans TimescaleDB
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class TimescaleDBService {

    private final JdbcTemplate jdbcTemplate;
    private final ObjectMapper objectMapper;

    private static final String INSERT_QUERY = """
        INSERT INTO raw_sensor_data 
        (time, asset_id, sensor_id, value, unit, quality, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?::jsonb)
        ON CONFLICT DO NOTHING
        """;

    /**
     * Insère une donnée capteur dans TimescaleDB
     */
    @Transactional
    public void insertSensorData(SensorData sensorData) {
        try {
            String metadataJson = convertMetadataToJson(sensorData.getMetadata());
            
            jdbcTemplate.update(INSERT_QUERY,
                Timestamp.from(sensorData.getTimestamp()),
                sensorData.getAssetId(),
                sensorData.getSensorId(),
                sensorData.getValue(),
                sensorData.getUnit(),
                sensorData.getQuality(),
                metadataJson
            );
            log.debug("Inserted sensor data: asset={}, sensor={}", 
                sensorData.getAssetId(), sensorData.getSensorId());
        } catch (Exception e) {
            log.error("Error inserting sensor data into TimescaleDB", e);
            throw new RuntimeException("Failed to insert into TimescaleDB", e);
        }
    }

    /**
     * Insère un batch de données
     */
    @Transactional
    public void insertBatch(List<SensorData> sensorDataList) {
        try {
            for (SensorData data : sensorDataList) {
                insertSensorData(data);
            }
            log.info("Inserted batch of {} records into TimescaleDB", sensorDataList.size());
        } catch (Exception e) {
            log.error("Error inserting batch into TimescaleDB", e);
            throw new RuntimeException("Failed to insert batch", e);
        }
    }

    private String convertMetadataToJson(Map<String, Object> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return "{}";
        }
        try {
            return objectMapper.writeValueAsString(metadata);
        } catch (Exception e) {
            log.warn("Failed to convert metadata to JSON, using empty object", e);
            return "{}";
        }
    }
}
