package com.predictivemaintenance.ingestion.service;

import com.predictivemaintenance.ingestion.model.SensorData;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@DisplayName("IngestionService Tests")
class IngestionServiceTest {

    @Mock
    private OPCUAService opcuaService;

    @Mock
    private DataNormalizationService normalizationService;

    @Mock
    private KafkaProducerService kafkaProducerService;

    @Mock
    private TimescaleDBService timescaleDBService;

    @Mock
    private MinIOService minioService;

    @InjectMocks
    private IngestionService ingestionService;

    private SensorData testSensorData;

    @BeforeEach
    void setUp() {
        testSensorData = SensorData.builder()
                .timestamp(Instant.now())
                .assetId("ASSET001")
                .sensorId("SENSOR001")
                .value(25.5)
                .unit("°C")
                .quality(2)
                .sourceType("OPC_UA")
                .build();
    }

    @Test
    @DisplayName("Should process sensor data through pipeline")
    void testProcessSensorData() {
        // Given
        SensorData normalized = SensorData.builder()
                .timestamp(Instant.now())
                .assetId("ASSET001")
                .sensorId("SENSOR001")
                .value(25.5)
                .unit("°C")
                .quality(2)
                .sourceType("OPC_UA")
                .build();

        when(normalizationService.normalize(any(SensorData.class))).thenReturn(normalized);
        doNothing().when(kafkaProducerService).publishSensorData(any(SensorData.class));
        doNothing().when(timescaleDBService).insertSensorData(any(SensorData.class));
        doNothing().when(minioService).storeSensorData(any(SensorData.class));

        // When
        ingestionService.processSensorData(testSensorData);

        // Then
        verify(normalizationService, times(1)).normalize(testSensorData);
        verify(kafkaProducerService, times(1)).publishSensorData(normalized);
        verify(timescaleDBService, times(1)).insertSensorData(normalized);
        verify(minioService, times(1)).storeSensorData(normalized);
    }

    @Test
    @DisplayName("Should handle normalization failure gracefully")
    void testProcessSensorDataWithNormalizationFailure() {
        // Given
        when(normalizationService.normalize(any(SensorData.class)))
                .thenThrow(new IllegalArgumentException("Invalid data"));

        // When/Then - Should not throw exception
        assertDoesNotThrow(() -> {
            ingestionService.processSensorData(testSensorData);
        });

        // Verify that downstream services are not called
        verify(kafkaProducerService, never()).publishSensorData(any());
        verify(timescaleDBService, never()).insertSensorData(any());
        verify(minioService, never()).storeSensorData(any());
    }

    @Test
    @DisplayName("Should handle null normalized data")
    void testProcessSensorDataWithNullNormalized() {
        // Given
        when(normalizationService.normalize(any(SensorData.class))).thenReturn(null);

        // When/Then - Should not throw exception
        assertDoesNotThrow(() -> {
            ingestionService.processSensorData(testSensorData);
        });
    }

    @Test
    @DisplayName("Should collect and process data from OPC UA")
    void testCollectAndProcessData() {
        // Given
        List<SensorData> rawDataList = new ArrayList<>();
        rawDataList.add(testSensorData);

        SensorData normalized = SensorData.builder()
                .timestamp(Instant.now())
                .assetId("ASSET001")
                .sensorId("SENSOR001")
                .value(25.5)
                .unit("°C")
                .quality(2)
                .sourceType("OPC_UA")
                .build();

        when(opcuaService.readAllNodes()).thenReturn(rawDataList);
        when(normalizationService.normalize(any(SensorData.class))).thenReturn(normalized);
        doNothing().when(kafkaProducerService).publishBatch(any());
        doNothing().when(timescaleDBService).insertBatch(any());
        doNothing().when(minioService).storeBatch(any());

        // When
        ingestionService.collectAndProcessData();

        // Then
        verify(opcuaService, times(1)).readAllNodes();
        verify(normalizationService, times(1)).normalize(any(SensorData.class));
        verify(kafkaProducerService, times(1)).publishBatch(any());
        verify(timescaleDBService, times(1)).insertBatch(any());
        verify(minioService, times(1)).storeBatch(any());
    }

    @Test
    @DisplayName("Should handle empty data list")
    void testCollectAndProcessDataWithEmptyList() {
        // Given
        when(opcuaService.readAllNodes()).thenReturn(new ArrayList<>());

        // When
        ingestionService.collectAndProcessData();

        // Then
        verify(opcuaService, times(1)).readAllNodes();
        verify(normalizationService, never()).normalize(any());
        verify(kafkaProducerService, never()).publishBatch(any());
    }
}

