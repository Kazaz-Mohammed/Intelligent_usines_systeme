package com.predictivemaintenance.orchestrateur.service;

import com.predictivemaintenance.orchestrateur.model.WorkOrder;
import com.predictivemaintenance.orchestrateur.model.WorkOrderStatus;
import com.predictivemaintenance.orchestrateur.repository.WorkOrderRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * Service pour la gestion des ordres de travail
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class WorkOrderService {
    
    private final WorkOrderRepository workOrderRepository;
    
    /**
     * Sauvegarde un ordre de travail
     */
    @Transactional
    public WorkOrder save(WorkOrder workOrder) {
        log.debug("Sauvegarde de l'ordre de travail: {}", workOrder.getWorkOrderNumber());
        return workOrderRepository.save(workOrder);
    }
    
    /**
     * Sauvegarde plusieurs ordres de travail
     */
    @Transactional
    public List<WorkOrder> saveAll(List<WorkOrder> workOrders) {
        log.debug("Sauvegarde de {} ordres de travail", workOrders.size());
        return workOrderRepository.saveAll(workOrders);
    }
    
    /**
     * Trouve un ordre de travail par ID
     */
    public Optional<WorkOrder> findById(Long id) {
        return workOrderRepository.findById(id);
    }
    
    /**
     * Trouve un ordre de travail par numéro
     */
    public Optional<WorkOrder> findByWorkOrderNumber(String workOrderNumber) {
        return workOrderRepository.findByWorkOrderNumber(workOrderNumber);
    }
    
    /**
     * Trouve tous les ordres de travail pour un actif
     */
    public List<WorkOrder> findByAssetId(String assetId) {
        return workOrderRepository.findByAssetIdOrderByScheduledStartTimeDesc(assetId);
    }
    
    /**
     * Trouve tous les ordres de travail par statut
     */
    public List<WorkOrder> findByStatus(WorkOrderStatus status) {
        return workOrderRepository.findByStatusOrderByScheduledStartTimeAsc(status);
    }
    
    /**
     * Met à jour le statut d'un ordre de travail
     */
    @Transactional
    public WorkOrder updateStatus(Long id, WorkOrderStatus newStatus) {
        WorkOrder workOrder = workOrderRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Ordre de travail non trouvé: " + id));
        
        workOrder.setStatus(newStatus);
        
        // Mettre à jour les temps réels si nécessaire
        if (newStatus == WorkOrderStatus.IN_PROGRESS && workOrder.getActualStartTime() == null) {
            workOrder.setActualStartTime(LocalDateTime.now());
        } else if (newStatus == WorkOrderStatus.COMPLETED && workOrder.getActualEndTime() == null) {
            workOrder.setActualEndTime(LocalDateTime.now());
            if (workOrder.getActualStartTime() != null) {
                long durationMinutes = java.time.Duration.between(
                        workOrder.getActualStartTime(),
                        workOrder.getActualEndTime()
                ).toMinutes();
                workOrder.setActualDurationMinutes((int) durationMinutes);
            }
        }
        
        return workOrderRepository.save(workOrder);
    }
    
    /**
     * Trouve tous les ordres de travail
     */
    public List<WorkOrder> findAll() {
        return workOrderRepository.findAll();
    }
    
    /**
     * Compte les ordres de travail par statut
     */
    public long countByStatus(WorkOrderStatus status) {
        return workOrderRepository.countByStatus(status);
    }
}

