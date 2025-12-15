package com.predictivemaintenance.ingestion.service;

import com.predictivemaintenance.ingestion.model.SensorData;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("DataNormalizationService Tests")
class DataNormalizationServiceTest {

    private DataNormalizationService normalizationService;

    @BeforeEach
    void setUp() {
        normalizationService = new DataNormalizationService();
    }

    @Test
    @DisplayName("Should normalize timestamp to UTC")
    void testNormalizeTimestamp() {
        SensorData rawData = SensorData.builder()
                .timestamp(Instant.now())
                .assetId("ASSET001")
                .sensorId("SENSOR001")
                .value(25.5)
                .unit("°C")
                .sourceType("OPC_UA")
                .build();

        SensorData normalized = normalizationService.normalize(rawData);

        assertNotNull(normalized.getTimestamp());
        assertEquals("ASSET001", normalized.getAssetId());
    }

    @Test
    @DisplayName("Should normalize null timestamp to current time")
    void testNormalizeNullTimestamp() {
        SensorData rawData = SensorData.builder()
                .timestamp(null)
                .assetId("ASSET001")
                .sensorId("SENSOR001")
                .value(25.5)
                .unit("°C")
                .sourceType("OPC_UA")
                .build();

        SensorData normalized = normalizationService.normalize(rawData);

        assertNotNull(normalized.getTimestamp());
    }

    @Test
    @DisplayName("Should normalize asset ID to uppercase")
    void testNormalizeAssetId() {
        SensorData rawData = SensorData.builder()
                .timestamp(Instant.now())
                .assetId("asset001")
                .sensorId("sensor001")
                .value(25.5)
                .unit("°C")
                .sourceType("OPC_UA")
                .build();

        SensorData normalized = normalizationService.normalize(rawData);

        assertEquals("ASSET001", normalized.getAssetId());
        assertEquals("SENSOR001", normalized.getSensorId());
    }

    @Test
    @DisplayName("Should convert Fahrenheit to Celsius")
    void testConvertFahrenheitToCelsius() {
        SensorData rawData = SensorData.builder()
                .timestamp(Instant.now())
                .assetId("ASSET001")
                .sensorId("SENSOR001")
                .value(77.0) // 77°F = 25°C
                .unit("°F")
                .sourceType("OPC_UA")
                .build();

        SensorData normalized = normalizationService.normalize(rawData);

        assertEquals(25.0, normalized.getValue(), 0.1);
        assertEquals("°C", normalized.getUnit());
    }

    @Test
    @DisplayName("Should normalize quality to valid range")
    void testNormalizeQuality() {
        SensorData rawData1 = SensorData.builder()
                .timestamp(Instant.now())
                .assetId("ASSET001")
                .sensorId("SENSOR001")
                .value(25.5)
                .unit("°C")
                .quality(5) // Out of range
                .sourceType("OPC_UA")
                .build();

        SensorData normalized1 = normalizationService.normalize(rawData1);
        assertEquals(2, normalized1.getQuality()); // Should be capped at 2

        SensorData rawData2 = SensorData.builder()
                .timestamp(Instant.now())
                .assetId("ASSET001")
                .sensorId("SENSOR001")
                .value(25.5)
                .unit("°C")
                .quality(null)
                .sourceType("OPC_UA")
                .build();

        SensorData normalized2 = normalizationService.normalize(rawData2);
        assertEquals(2, normalized2.getQuality()); // Default to 2 (good)
    }

    @Test
    @DisplayName("Should normalize metadata")
    void testNormalizeMetadata() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("source", "test");

        SensorData rawData = SensorData.builder()
                .timestamp(Instant.now())
                .assetId("ASSET001")
                .sensorId("SENSOR001")
                .value(25.5)
                .unit("°C")
                .metadata(metadata)
                .sourceType("OPC_UA")
                .build();

        SensorData normalized = normalizationService.normalize(rawData);

        assertNotNull(normalized.getMetadata());
        assertTrue(normalized.getMetadata().containsKey("source"));
        assertTrue(normalized.getMetadata().containsKey("normalized_at"));
    }

    @Test
    @DisplayName("Should throw exception for null asset ID")
    void testNullAssetId() {
        SensorData rawData = SensorData.builder()
                .timestamp(Instant.now())
                .assetId(null)
                .sensorId("SENSOR001")
                .value(25.5)
                .unit("°C")
                .sourceType("OPC_UA")
                .build();

        assertThrows(IllegalArgumentException.class, () -> {
            normalizationService.normalize(rawData);
        });
    }

    @Test
    @DisplayName("Should throw exception for null sensor ID")
    void testNullSensorId() {
        SensorData rawData = SensorData.builder()
                .timestamp(Instant.now())
                .assetId("ASSET001")
                .sensorId(null)
                .value(25.5)
                .unit("°C")
                .sourceType("OPC_UA")
                .build();

        assertThrows(IllegalArgumentException.class, () -> {
            normalizationService.normalize(rawData);
        });
    }

    @Test
    @DisplayName("Should throw exception for null value")
    void testNullValue() {
        SensorData rawData = SensorData.builder()
                .timestamp(Instant.now())
                .assetId("ASSET001")
                .sensorId("SENSOR001")
                .value(null)
                .unit("°C")
                .sourceType("OPC_UA")
                .build();

        assertThrows(IllegalArgumentException.class, () -> {
            normalizationService.normalize(rawData);
        });
    }

    @Test
    @DisplayName("Should return null for null input")
    void testNullInput() {
        SensorData normalized = normalizationService.normalize(null);
        assertNull(normalized);
    }
}

