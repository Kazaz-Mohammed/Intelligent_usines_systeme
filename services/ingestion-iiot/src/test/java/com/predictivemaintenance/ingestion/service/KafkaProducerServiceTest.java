package com.predictivemaintenance.ingestion.service;

import com.predictivemaintenance.ingestion.model.SensorData;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.util.concurrent.SettableListenableFuture;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@DisplayName("KafkaProducerService Tests")
class KafkaProducerServiceTest {

    @Mock
    private KafkaTemplate<String, Object> kafkaTemplate;

    @InjectMocks
    private KafkaProducerService kafkaProducerService;

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
    @DisplayName("Should publish sensor data to Kafka")
    void testPublishSensorData() {
        // Given
        CompletableFuture<SendResult<String, Object>> future = new CompletableFuture<>();
        SendResult<String, Object> sendResult = mock(SendResult.class);
        future.complete(sendResult);

        when(kafkaTemplate.send(anyString(), anyString(), any(SensorData.class)))
                .thenReturn(future);

        // When
        kafkaProducerService.publishSensorData(testSensorData);

        // Then
        ArgumentCaptor<String> topicCaptor = ArgumentCaptor.forClass(String.class);
        ArgumentCaptor<String> keyCaptor = ArgumentCaptor.forClass(String.class);
        ArgumentCaptor<SensorData> valueCaptor = ArgumentCaptor.forClass(SensorData.class);

        verify(kafkaTemplate, times(1)).send(
                topicCaptor.capture(),
                keyCaptor.capture(),
                valueCaptor.capture()
        );

        assertEquals("sensor-data", topicCaptor.getValue());
        assertEquals("ASSET001:SENSOR001", keyCaptor.getValue());
        assertEquals(testSensorData, valueCaptor.getValue());
    }

    @Test
    @DisplayName("Should publish batch of sensor data")
    void testPublishBatch() {
        // Given
        List<SensorData> dataList = new ArrayList<>();
        dataList.add(testSensorData);
        dataList.add(SensorData.builder()
                .timestamp(Instant.now())
                .assetId("ASSET002")
                .sensorId("SENSOR002")
                .value(30.0)
                .unit("°C")
                .sourceType("OPC_UA")
                .build());

        CompletableFuture<SendResult<String, Object>> future = new CompletableFuture<>();
        future.complete(mock(SendResult.class));

        when(kafkaTemplate.send(anyString(), anyString(), any(SensorData.class)))
                .thenReturn(future);

        // When
        kafkaProducerService.publishBatch(dataList);

        // Then
        verify(kafkaTemplate, times(2)).send(anyString(), anyString(), any(SensorData.class));
    }

    @Test
    @DisplayName("Should handle Kafka send failure gracefully")
    void testPublishFailure() {
        // Given
        CompletableFuture<SendResult<String, Object>> future = new CompletableFuture<>();
        future.completeExceptionally(new RuntimeException("Kafka error"));

        when(kafkaTemplate.send(anyString(), anyString(), any(SensorData.class)))
                .thenReturn(future);

        // When/Then - Should not throw exception, just log error
        assertDoesNotThrow(() -> {
            kafkaProducerService.publishSensorData(testSensorData);
        });

        // Wait a bit for async completion
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}

