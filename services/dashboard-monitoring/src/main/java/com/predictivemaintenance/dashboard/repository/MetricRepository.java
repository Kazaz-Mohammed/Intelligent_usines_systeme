package com.predictivemaintenance.dashboard.repository;

import com.predictivemaintenance.dashboard.model.Metric;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * Repository pour les métriques
 */
@Repository
public interface MetricRepository extends JpaRepository<Metric, Long> {
    
    List<Metric> findByMetricNameAndTimestampBetweenOrderByTimestampAsc(
            String metricName, LocalDateTime start, LocalDateTime end
    );
    
    List<Metric> findByServiceNameAndTimestampBetweenOrderByTimestampAsc(
            String serviceName, LocalDateTime start, LocalDateTime end
    );
    
    @Query(value = "SELECT * FROM metrics WHERE metric_name = :metricName ORDER BY \"timestamp\" DESC LIMIT 1", nativeQuery = true)
    Optional<Metric> findTopByMetricNameOrderByTimestampDesc(@Param("metricName") String metricName);
    
    @Query(value = "SELECT * FROM metrics ORDER BY \"timestamp\" DESC LIMIT :limit", nativeQuery = true)
    List<Metric> findTopNByOrderByTimestampDesc(@Param("limit") int limit);
}

