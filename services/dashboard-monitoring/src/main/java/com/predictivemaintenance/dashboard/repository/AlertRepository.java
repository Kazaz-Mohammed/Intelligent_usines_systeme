package com.predictivemaintenance.dashboard.repository;

import com.predictivemaintenance.dashboard.model.Alert;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * Repository pour les alertes
 */
@Repository
public interface AlertRepository extends JpaRepository<Alert, Long> {
    
    List<Alert> findByStatusOrderByCreatedAtDesc(Alert.AlertStatus status);
    
    List<Alert> findBySeverityOrderByCreatedAtDesc(Alert.Severity severity);
    
    List<Alert> findByAssetIdOrderByCreatedAtDesc(String assetId);
    
    long countByStatus(Alert.AlertStatus status);
    
    long countBySeverity(Alert.Severity severity);
}

