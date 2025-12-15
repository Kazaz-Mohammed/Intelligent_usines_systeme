package com.predictivemaintenance.dashboard.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictivemaintenance.dashboard.model.DashboardOverview;
import com.predictivemaintenance.dashboard.service.DashboardService;
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
 * Tests pour DashboardController
 */
@WebMvcTest(DashboardController.class)
class DashboardControllerTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    @MockBean
    private DashboardService dashboardService;
    
    @Autowired
    private ObjectMapper objectMapper;
    
    @Test
    void testGetOverview() throws Exception {
        DashboardOverview overview = DashboardOverview.builder()
                .totalAssets(100L)
                .assetsWithAnomalies(15L)
                .activeInterventions(8L)
                .build();
        
        when(dashboardService.getOverview()).thenReturn(overview);
        
        mockMvc.perform(get("/api/v1/dashboard/overview"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.totalAssets").value(100))
                .andExpect(jsonPath("$.assetsWithAnomalies").value(15))
                .andExpect(jsonPath("$.activeInterventions").value(8));
    }
    
    @Test
    void testGetRealtimeMetrics() throws Exception {
        Map<String, Object> metrics = Map.of(
                "services", Map.of(),
                "system", Map.of("cpuUsage", 75.5),
                "maintenance", Map.of()
        );
        
        when(dashboardService.getRealtimeMetrics()).thenReturn(metrics);
        
        mockMvc.perform(get("/api/v1/dashboard/metrics"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.system.cpuUsage").value(75.5));
    }
    
    @Test
    void testGetStatistics() throws Exception {
        mockMvc.perform(get("/api/v1/dashboard/statistics"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.message").exists());
    }
}

