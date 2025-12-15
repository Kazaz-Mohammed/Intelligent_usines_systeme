package com.predictivemaintenance.ingestion.service;

import com.predictivemaintenance.ingestion.config.OPCUAConfig;
import com.predictivemaintenance.ingestion.model.SensorData;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@DisplayName("OPCUAService Tests")
class OPCUAServiceTest {

    @Mock
    private OPCUAConfig opcuaConfig;

    @InjectMocks
    private OPCUAService opcuaService;

    @BeforeEach
    void setUp() {
        when(opcuaConfig.isEnabled()).thenReturn(true);
        when(opcuaConfig.getEndpointUrl()).thenReturn("opc.tcp://localhost:4840");
    }

    @Test
    @DisplayName("Should return empty list when OPC UA is disabled")
    void testReadAllNodesWhenDisabled() {
        // Given
        when(opcuaConfig.isEnabled()).thenReturn(false);
        when(opcuaConfig.getNodes()).thenReturn(null);

        // When
        List<SensorData> result = opcuaService.readAllNodes();

        // Then
        assertNotNull(result);
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("Should return empty list when no nodes configured")
    void testReadAllNodesWhenNoNodes() {
        // Given
        when(opcuaConfig.getNodes()).thenReturn(null);

        // When
        List<SensorData> result = opcuaService.readAllNodes();

        // Then
        assertNotNull(result);
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("Should handle connection failure gracefully")
    void testConnectFailure() {
        // When/Then - Should throw exception but handle gracefully
        // Note: This test may need adjustment based on actual implementation
        assertThrows(Exception.class, () -> {
            opcuaService.connect();
        });
    }
}

