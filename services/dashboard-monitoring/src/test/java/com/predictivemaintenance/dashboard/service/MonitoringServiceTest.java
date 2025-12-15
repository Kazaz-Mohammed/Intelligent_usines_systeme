package com.predictivemaintenance.dashboard.service;

import com.predictivemaintenance.dashboard.config.DashboardConfig;
import com.predictivemaintenance.dashboard.model.DashboardOverview;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/**
 * Tests pour MonitoringService
 */
class MonitoringServiceTest {
    
    @Mock
    private DashboardConfig config;
    
    @Mock
    private WebClient.Builder webClientBuilder;
    
    private MonitoringService monitoringService;
    
    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        
        DashboardConfig.Monitoring monitoring = new DashboardConfig.Monitoring();
        monitoring.setServices(Map.of(
                "test-service", "http://localhost:8080"
        ));
        monitoring.setUpdateIntervalSeconds("5");
        
        when(config.getMonitoring()).thenReturn(monitoring);
        when(config.getServices()).thenReturn(monitoring.getServices());
        when(config.getUpdateIntervalSeconds()).thenReturn("5");
        
        monitoringService = new MonitoringService(config, webClientBuilder);
    }
    
    @Test
    void testGetAllServiceMetrics() {
        Map<String, DashboardOverview.ServiceMetrics> metrics = 
                monitoringService.getAllServiceMetrics();
        
        assertNotNull(metrics);
    }
    
    @Test
    void testGetServiceMetrics() {
        DashboardOverview.ServiceMetrics metrics = 
                monitoringService.getServiceMetrics("unknown-service");
        
        assertNotNull(metrics);
        assertEquals("unknown-service", metrics.getServiceName());
        assertEquals(DashboardOverview.ServiceMetrics.ServiceStatus.UNKNOWN, 
                metrics.getStatus());
    }
}

