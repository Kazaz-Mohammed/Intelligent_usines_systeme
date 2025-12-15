package com.predictivemaintenance.orchestrateur.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.predictivemaintenance.orchestrateur.model.InterventionRequest;
import com.predictivemaintenance.orchestrateur.model.PriorityLevel;
import com.predictivemaintenance.orchestrateur.model.WorkOrder;
import com.predictivemaintenance.orchestrateur.model.WorkOrderStatus;
import com.predictivemaintenance.orchestrateur.service.PlanningService;
import com.predictivemaintenance.orchestrateur.service.WorkOrderService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDateTime;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * Tests pour InterventionController
 */
@WebMvcTest(InterventionController.class)
class InterventionControllerTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    @MockBean
    private PlanningService planningService;
    
    @MockBean
    private WorkOrderService workOrderService;
    
    @Autowired
    private ObjectMapper objectMapper;
    
    @Test
    void testCreateIntervention() throws Exception {
        InterventionRequest request = InterventionRequest.builder()
                .assetId("ASSET001")
                .priority(PriorityLevel.HIGH)
                .interventionType("corrective")
                .build();
        
        WorkOrder workOrder = WorkOrder.builder()
                .workOrderNumber("WO-001")
                .assetId("ASSET001")
                .priority(PriorityLevel.HIGH)
                .status(WorkOrderStatus.SCHEDULED)
                .scheduledStartTime(LocalDateTime.now().plusHours(4))
                .build();
        
        when(planningService.planIntervention(any(InterventionRequest.class)))
                .thenReturn(workOrder);
        when(workOrderService.save(any(WorkOrder.class))).thenReturn(workOrder);
        
        mockMvc.perform(post("/api/v1/interventions")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.workOrderNumber").value("WO-001"))
                .andExpect(jsonPath("$.assetId").value("ASSET001"));
    }
    
    @Test
    void testCreateInterventionsBatch() throws Exception {
        List<InterventionRequest> requests = List.of(
                InterventionRequest.builder()
                        .assetId("ASSET001")
                        .priority(PriorityLevel.HIGH)
                        .build(),
                InterventionRequest.builder()
                        .assetId("ASSET002")
                        .priority(PriorityLevel.MEDIUM)
                        .build()
        );
        
        List<WorkOrder> workOrders = List.of(
                WorkOrder.builder()
                        .workOrderNumber("WO-001")
                        .assetId("ASSET001")
                        .build(),
                WorkOrder.builder()
                        .workOrderNumber("WO-002")
                        .assetId("ASSET002")
                        .build()
        );
        
        when(planningService.planInterventionsOptimized(any()))
                .thenReturn(workOrders);
        when(workOrderService.saveAll(any())).thenReturn(workOrders);
        
        mockMvc.perform(post("/api/v1/interventions/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(requests)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$").isArray())
                .andExpect(jsonPath("$.length()").value(2));
    }
}

