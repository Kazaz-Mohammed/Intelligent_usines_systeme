package com.predictivemaintenance.orchestrateur.repository;

import com.predictivemaintenance.orchestrateur.model.PriorityLevel;
import com.predictivemaintenance.orchestrateur.model.WorkOrder;
import com.predictivemaintenance.orchestrateur.model.WorkOrderStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * Repository JPA pour les ordres de travail
 */
@Repository
public interface WorkOrderRepository extends JpaRepository<WorkOrder, Long> {
    
    /**
     * Trouve un ordre de travail par son numéro
     */
    Optional<WorkOrder> findByWorkOrderNumber(String workOrderNumber);
    
    /**
     * Trouve tous les ordres de travail pour un actif
     */
    List<WorkOrder> findByAssetIdOrderByScheduledStartTimeDesc(String assetId);
    
    /**
     * Trouve tous les ordres de travail par statut
     */
    List<WorkOrder> findByStatusOrderByScheduledStartTimeAsc(WorkOrderStatus status);
    
    /**
     * Trouve tous les ordres de travail par priorité
     */
    List<WorkOrder> findByPriorityOrderByScheduledStartTimeAsc(PriorityLevel priority);
    
    /**
     * Trouve tous les ordres de travail planifiés dans une période
     */
    @Query("SELECT wo FROM WorkOrder wo WHERE wo.scheduledStartTime >= :start AND wo.scheduledStartTime <= :end ORDER BY wo.scheduledStartTime ASC")
    List<WorkOrder> findScheduledBetween(
            @Param("start") LocalDateTime start,
            @Param("end") LocalDateTime end
    );
    
    /**
     * Trouve tous les ordres de travail en cours
     */
    @Query("SELECT wo FROM WorkOrder wo WHERE wo.status = 'IN_PROGRESS' ORDER BY wo.scheduledStartTime ASC")
    List<WorkOrder> findInProgress();
    
    /**
     * Compte les ordres de travail par statut
     */
    long countByStatus(WorkOrderStatus status);
    
    /**
     * Compte les ordres de travail par priorité
     */
    long countByPriority(PriorityLevel priority);
}

