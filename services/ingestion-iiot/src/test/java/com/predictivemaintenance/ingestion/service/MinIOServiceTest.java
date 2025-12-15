package com.predictivemaintenance.ingestion.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictivemaintenance.ingestion.model.SensorData;
import io.minio.MinioClient;
import io.minio.PutObjectArgs;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.ByteArrayInputStream;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@DisplayName("MinIOService Tests")
class MinIOServiceTest {

    @Mock
    private MinioClient minioClient;

    @Mock
    private ObjectMapper objectMapper;

    @InjectMocks
    private MinIOService minioService;

    private SensorData testSensorData;

    @BeforeEach
    void setUp() throws Exception {
        testSensorData = SensorData.builder()
                .timestamp(Instant.now())
                .assetId("ASSET001")
                .sensorId("SENSOR001")
                .value(25.5)
                .unit("°C")
                .quality(2)
                .sourceType("OPC_UA")
                .build();

        when(objectMapper.writeValueAsString(any(SensorData.class))).thenReturn("{\"test\":\"data\"}");
    }

    @Test
    @DisplayName("Should store sensor data in MinIO")
    void testStoreSensorData() throws Exception {
        // Given
        doNothing().when(minioClient).putObject(any(PutObjectArgs.class));

        // When
        minioService.storeSensorData(testSensorData);

        // Then
        ArgumentCaptor<PutObjectArgs> argsCaptor = ArgumentCaptor.forClass(PutObjectArgs.class);
        verify(minioClient, times(1)).putObject(argsCaptor.capture());

        PutObjectArgs args = argsCaptor.getValue();
        assertEquals("raw-sensor-data", args.bucket());
        assertTrue(args.object().contains("ASSET001"));
        assertTrue(args.object().contains("SENSOR001"));
    }

    @Test
    @DisplayName("Should store batch of sensor data")
    void testStoreBatch() throws Exception {
        // Given
        List<SensorData> dataList = new ArrayList<>();
        dataList.add(testSensorData);
        dataList.add(SensorData.builder()
                .timestamp(Instant.now())
                .assetId("ASSET002")
                .sensorId("SENSOR002")
                .value(30.0)
                .unit("°C")
                .quality(2)
                .sourceType("OPC_UA")
                .build());

        doNothing().when(minioClient).putObject(any(PutObjectArgs.class));

        // When
        minioService.storeBatch(dataList);

        // Then
        verify(minioClient, times(2)).putObject(any(PutObjectArgs.class));
    }

    @Test
    @DisplayName("Should handle MinIO error")
    void testStoreWithMinIOError() throws Exception {
        // Given
        doThrow(new RuntimeException("MinIO error")).when(minioClient)
                .putObject(any(PutObjectArgs.class));

        // When/Then
        assertThrows(RuntimeException.class, () -> {
            minioService.storeSensorData(testSensorData);
        });
    }
}

