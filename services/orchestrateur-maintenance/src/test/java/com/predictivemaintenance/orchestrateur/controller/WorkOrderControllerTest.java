package com.predictivemaintenance.orchestrateur.controller;

import com.predictivemaintenance.orchestrateur.model.PriorityLevel;
import com.predictivemaintenance.orchestrateur.model.WorkOrder;
import com.predictivemaintenance.orchestrateur.model.WorkOrderStatus;
import com.predictivemaintenance.orchestrateur.service.WorkOrderService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * Tests pour WorkOrderController
 */
@WebMvcTest(WorkOrderController.class)
class WorkOrderControllerTest {
    
    @Autowired
    private MockMvc mockMvc;
    
    @MockBean
    private WorkOrderService workOrderService;
    
    @Test
    void testGetAllWorkOrders() throws Exception {
        List<WorkOrder> workOrders = List.of(
                WorkOrder.builder().workOrderNumber("WO-001").build(),
                WorkOrder.builder().workOrderNumber("WO-002").build()
        );
        
        when(workOrderService.findAll()).thenReturn(workOrders);
        
        mockMvc.perform(get("/api/v1/work-orders"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray())
                .andExpect(jsonPath("$.length()").value(2));
    }
    
    @Test
    void testGetWorkOrderById() throws Exception {
        WorkOrder workOrder = WorkOrder.builder()
                .id(1L)
                .workOrderNumber("WO-001")
                .assetId("ASSET001")
                .build();
        
        when(workOrderService.findById(1L)).thenReturn(Optional.of(workOrder));
        
        mockMvc.perform(get("/api/v1/work-orders/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.workOrderNumber").value("WO-001"))
                .andExpect(jsonPath("$.assetId").value("ASSET001"));
    }
    
    @Test
    void testGetWorkOrderByIdNotFound() throws Exception {
        when(workOrderService.findById(1L)).thenReturn(Optional.empty());
        
        mockMvc.perform(get("/api/v1/work-orders/1"))
                .andExpect(status().isNotFound());
    }
    
    @Test
    void testGetWorkOrderByNumber() throws Exception {
        WorkOrder workOrder = WorkOrder.builder()
                .workOrderNumber("WO-001")
                .build();
        
        when(workOrderService.findByWorkOrderNumber("WO-001"))
                .thenReturn(Optional.of(workOrder));
        
        mockMvc.perform(get("/api/v1/work-orders/number/WO-001"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.workOrderNumber").value("WO-001"));
    }
    
    @Test
    void testGetWorkOrdersByAsset() throws Exception {
        List<WorkOrder> workOrders = List.of(
                WorkOrder.builder().assetId("ASSET001").build()
        );
        
        when(workOrderService.findByAssetId("ASSET001")).thenReturn(workOrders);
        
        mockMvc.perform(get("/api/v1/work-orders/asset/ASSET001"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray())
                .andExpect(jsonPath("$[0].assetId").value("ASSET001"));
    }
    
    @Test
    void testGetWorkOrdersByStatus() throws Exception {
        List<WorkOrder> workOrders = List.of(
                WorkOrder.builder()
                        .status(WorkOrderStatus.SCHEDULED)
                        .build()
        );
        
        when(workOrderService.findByStatus(WorkOrderStatus.SCHEDULED))
                .thenReturn(workOrders);
        
        mockMvc.perform(get("/api/v1/work-orders/status/SCHEDULED"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray())
                .andExpect(jsonPath("$[0].status").value("SCHEDULED"));
    }
    
    @Test
    void testUpdateWorkOrderStatus() throws Exception {
        WorkOrder workOrder = WorkOrder.builder()
                .id(1L)
                .status(WorkOrderStatus.IN_PROGRESS)
                .build();
        
        when(workOrderService.updateStatus(1L, WorkOrderStatus.IN_PROGRESS))
                .thenReturn(workOrder);
        
        mockMvc.perform(put("/api/v1/work-orders/1/status")
                        .contentType("application/json")
                        .content("{\"status\":\"IN_PROGRESS\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("IN_PROGRESS"));
    }
    
    @Test
    void testGetWorkOrderStats() throws Exception {
        when(workOrderService.findAll()).thenReturn(List.of(
                WorkOrder.builder().status(WorkOrderStatus.SCHEDULED).build(),
                WorkOrder.builder().status(WorkOrderStatus.IN_PROGRESS).build()
        ));
        when(workOrderService.countByStatus(WorkOrderStatus.SCHEDULED)).thenReturn(1L);
        when(workOrderService.countByStatus(WorkOrderStatus.IN_PROGRESS)).thenReturn(1L);
        when(workOrderService.countByStatus(WorkOrderStatus.COMPLETED)).thenReturn(0L);
        when(workOrderService.countByStatus(WorkOrderStatus.CANCELLED)).thenReturn(0L);
        when(workOrderService.countByStatus(WorkOrderStatus.PENDING)).thenReturn(0L);
        
        mockMvc.perform(get("/api/v1/work-orders/stats"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.total").exists())
                .andExpect(jsonPath("$.scheduled").exists())
                .andExpect(jsonPath("$.inProgress").exists());
    }
}

