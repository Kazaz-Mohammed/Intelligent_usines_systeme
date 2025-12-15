package com.predictivemaintenance.ingestion.integration;

import com.predictivemaintenance.ingestion.model.SensorData;
import com.predictivemaintenance.ingestion.service.IngestionService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.KafkaContainer;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@Testcontainers
@ActiveProfiles("test")
@DisplayName("Ingestion Integration Tests")
class IngestionIntegrationTest {

    @Container
    static KafkaContainer kafka = new KafkaContainer(
            DockerImageName.parse("confluentinc/cp-kafka:7.5.0"));

    @Container
    static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>(
            DockerImageName.parse("timescale/timescaledb:latest-pg16"))
            .withDatabaseName("testdb")
            .withUsername("test")
            .withPassword("test");

    @DynamicPropertySource
    static void configureProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.kafka.bootstrap-servers", kafka::getBootstrapServers);
        registry.add("spring.datasource.url", postgres::getJdbcUrl);
        registry.add("spring.datasource.username", postgres::getUsername);
        registry.add("spring.datasource.password", postgres::getPassword);
    }

    @Autowired
    private IngestionService ingestionService;

    @Autowired(required = false)
    private KafkaTemplate<String, Object> kafkaTemplate;

    @Test
    @DisplayName("Should process sensor data end-to-end")
    void testProcessSensorData() {
        // Given
        SensorData sensorData = SensorData.builder()
                .timestamp(Instant.now())
                .assetId("ASSET001")
                .sensorId("SENSOR001")
                .value(25.5)
                .unit("°C")
                .quality(2)
                .metadata(new HashMap<>())
                .sourceType("TEST")
                .build();

        // When
        assertDoesNotThrow(() -> {
            ingestionService.processSensorData(sensorData);
        });

        // Then - Verify no exceptions thrown
        // In a real integration test, you would verify:
        // - Message published to Kafka
        // - Data inserted in PostgreSQL
        // - Data stored in MinIO
    }

    @Test
    @DisplayName("Should handle invalid sensor data gracefully")
    void testProcessInvalidSensorData() {
        // Given - Invalid data (null asset ID)
        SensorData invalidData = SensorData.builder()
                .timestamp(Instant.now())
                .assetId(null)
                .sensorId("SENSOR001")
                .value(25.5)
                .unit("°C")
                .sourceType("TEST")
                .build();

        // When/Then - Should handle error gracefully
        assertDoesNotThrow(() -> {
            ingestionService.processSensorData(invalidData);
        });
    }
}

