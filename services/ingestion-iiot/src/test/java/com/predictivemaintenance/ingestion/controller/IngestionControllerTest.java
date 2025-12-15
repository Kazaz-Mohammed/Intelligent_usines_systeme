package com.predictivemaintenance.ingestion.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictivemaintenance.ingestion.model.SensorData;
import com.predictivemaintenance.ingestion.service.IngestionService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;

import java.time.Instant;
import java.util.Map;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(IngestionController.class)
@DisplayName("IngestionController Tests")
class IngestionControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private IngestionService ingestionService;

    @Autowired
    private ObjectMapper objectMapper;

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
    @DisplayName("Should return health status")
    void testHealth() throws Exception {
        mockMvc.perform(get("/api/v1/ingestion/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("UP"))
                .andExpect(jsonPath("$.service").value("ingestion-iiot"));
    }

    @Test
    @DisplayName("Should return service status")
    void testStatus() throws Exception {
        mockMvc.perform(get("/api/v1/ingestion/status"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.service").value("ingestion-iiot"))
                .andExpect(jsonPath("$.status").value("running"));
    }

    @Test
    @DisplayName("Should ingest data successfully")
    void testIngestData() throws Exception {
        mockMvc.perform(post("/api/v1/ingestion/data")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(testSensorData)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("success"));

        // Verify service was called
        // Note: In a real test, you might want to verify the service call
    }

    @Test
    @DisplayName("Should handle ingestion error")
    void testIngestDataError() throws Exception {
        // Given - service throws exception
        doThrow(new RuntimeException("Processing error"))
                .when(ingestionService).processSensorData(any(SensorData.class));

        mockMvc.perform(post("/api/v1/ingestion/data")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(testSensorData)))
                .andExpect(status().isInternalServerError())
                .andExpect(jsonPath("$.status").value("error"));
    }
}

