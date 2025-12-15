package com.predictivemaintenance.ingestion.service;

import com.predictivemaintenance.ingestion.model.SensorData;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

/**
 * Service de normalisation des données capteurs
 * - Horodatage unifié (UTC)
 * - Conversion d'unités
 * - Gestion QoS
 */
@Slf4j
@Service
public class DataNormalizationService {

    /**
     * Normalise les données capteurs
     */
    public SensorData normalize(SensorData rawData) {
        if (rawData == null) {
            return null;
        }

        SensorData normalized = SensorData.builder()
                .timestamp(normalizeTimestamp(rawData.getTimestamp()))
                .assetId(normalizeAssetId(rawData.getAssetId()))
                .sensorId(normalizeSensorId(rawData.getSensorId()))
                .value(normalizeValue(rawData.getValue(), rawData.getUnit()))
                .unit(normalizeUnit(rawData.getUnit()))
                .quality(normalizeQuality(rawData.getQuality()))
                .metadata(normalizeMetadata(rawData.getMetadata()))
                .sourceType(rawData.getSourceType())
                .sourceEndpoint(rawData.getSourceEndpoint())
                .build();

        log.debug("Normalized data: {} -> {}", rawData, normalized);
        return normalized;
    }

    private Instant normalizeTimestamp(Instant timestamp) {
        if (timestamp == null) {
            return Instant.now();
        }
        // S'assurer que le timestamp est en UTC
        return timestamp;
    }

    private String normalizeAssetId(String assetId) {
        if (assetId == null || assetId.trim().isEmpty()) {
            throw new IllegalArgumentException("Asset ID cannot be null or empty");
        }
        return assetId.trim().toUpperCase();
    }

    private String normalizeSensorId(String sensorId) {
        if (sensorId == null || sensorId.trim().isEmpty()) {
            throw new IllegalArgumentException("Sensor ID cannot be null or empty");
        }
        return sensorId.trim().toUpperCase();
    }

    private Double normalizeValue(Double value, String unit) {
        if (value == null) {
            throw new IllegalArgumentException("Value cannot be null");
        }
        
        // Conversion d'unités si nécessaire
        // Exemple: convertir Fahrenheit en Celsius
        if ("°F".equals(unit) || "F".equals(unit)) {
            return (value - 32) * 5.0 / 9.0;
        }
        
        return value;
    }

    private String normalizeUnit(String unit) {
        if (unit == null || unit.trim().isEmpty()) {
            return "N/A";
        }
        
        // Normalisation des unités
        String normalized = unit.trim();
        if ("F".equals(normalized)) {
            return "°C"; // Converti en Celsius
        }
        
        return normalized;
    }

    private Integer normalizeQuality(Integer quality) {
        if (quality == null) {
            return 2; // Good par défaut
        }
        
        // S'assurer que quality est entre 0 et 2
        if (quality < 0) {
            return 0; // Bad
        }
        if (quality > 2) {
            return 2; // Good
        }
        return quality;
    }

    private Map<String, Object> normalizeMetadata(Map<String, Object> metadata) {
        if (metadata == null) {
            return new HashMap<>();
        }
        
        Map<String, Object> normalized = new HashMap<>(metadata);
        normalized.put("normalized_at", Instant.now().toString());
        return normalized;
    }
}

