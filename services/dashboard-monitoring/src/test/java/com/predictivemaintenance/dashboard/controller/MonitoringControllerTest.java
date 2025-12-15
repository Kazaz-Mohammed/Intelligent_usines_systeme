package com.predictivemaintenance.dashboard.controller;

import com.predictivemaintenance.dashboard.model.DashboardOverview;
import com.predictivemaintenance.dashboard.service.MonitoringService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.web.servlet.MockMvc;

import java.util.Map;

import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * Tests pour MonitoringController
 */
@WebMvcTest(MonitoringController.class)
class MonitoringControllerTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    @MockBean
    private MonitoringService monitoringService;
    
    @Test
    void testGetServicesStatus() throws Exception {
        Map<String, DashboardOverview.ServiceMetrics> services = Map.of(
                "test-service", DashboardOverview.ServiceMetrics.builder()
                        .serviceName("test-service")
                        .status(DashboardOverview.ServiceMetrics.ServiceStatus.UP)
                        .build()
        );
        
        when(monitoringService.getAllServiceMetrics()).thenReturn(services);
        
        mockMvc.perform(get("/api/v1/monitoring/services"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.test-service.status").value("UP"));
    }
    
    @Test
    void testGetServiceStatus() throws Exception {
        DashboardOverview.ServiceMetrics metrics = DashboardOverview.ServiceMetrics.builder()
                .serviceName("test-service")
                .status(DashboardOverview.ServiceMetrics.ServiceStatus.UP)
                .build();
        
        when(monitoringService.getServiceMetrics("test-service")).thenReturn(metrics);
        
        mockMvc.perform(get("/api/v1/monitoring/services/test-service"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.serviceName").value("test-service"))
                .andExpect(jsonPath("$.status").value("UP"));
    }
    
    @Test
    void testGetGlobalHealth() throws Exception {
        Map<String, DashboardOverview.ServiceMetrics> services = Map.of(
                "service1", DashboardOverview.ServiceMetrics.builder()
                        .status(DashboardOverview.ServiceMetrics.ServiceStatus.UP)
                        .build(),
                "service2", DashboardOverview.ServiceMetrics.builder()
                        .status(DashboardOverview.ServiceMetrics.ServiceStatus.UP)
                        .build()
        );
        
        when(monitoringService.getAllServiceMetrics()).thenReturn(services);
        
        mockMvc.perform(get("/api/v1/monitoring/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("HEALTHY"))
                .andExpect(jsonPath("$.upServices").value(2))
                .andExpect(jsonPath("$.totalServices").value(2));
    }
}

