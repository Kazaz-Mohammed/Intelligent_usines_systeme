package com.predictivemaintenance.ingestion.service;

import com.predictivemaintenance.ingestion.config.OPCUAConfig;
import com.predictivemaintenance.ingestion.model.SensorData;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.milo.opcua.sdk.client.OpcUaClient;
import org.eclipse.milo.opcua.sdk.client.api.config.OpcUaClientConfig;
import org.eclipse.milo.opcua.sdk.client.api.identity.AnonymousProvider;
import org.eclipse.milo.opcua.stack.client.DiscoveryClient;
import org.eclipse.milo.opcua.stack.core.UaException;
import org.eclipse.milo.opcua.stack.core.types.builtin.*;
import org.eclipse.milo.opcua.stack.core.types.builtin.unsigned.UInteger;
import org.eclipse.milo.opcua.stack.core.types.enumerated.TimestampsToReturn;
import org.eclipse.milo.opcua.stack.core.types.structured.EndpointDescription;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

/**
 * Service pour collecter les données depuis OPC UA
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class OPCUAService {

    private final OPCUAConfig opcuaConfig;
    private OpcUaClient client;

    /**
     * Se connecte au serveur OPC UA
     */
    public void connect() {
        if (!opcuaConfig.isEnabled()) {
            log.info("OPC UA is disabled");
            return;
        }

        try {
            List<EndpointDescription> endpoints = DiscoveryClient.getEndpoints(opcuaConfig.getEndpointUrl())
                    .get();

            EndpointDescription endpoint = endpoints.get(0);

            OpcUaClientConfig config = OpcUaClientConfig.builder()
                    .setEndpoint(endpoint)
                    .setIdentityProvider(new AnonymousProvider())
                    .setRequestTimeout(UInteger.valueOf(opcuaConfig.getRequestTimeout()))
                    .build();

            client = OpcUaClient.create(config);
            client.connect().get();

            log.info("Connected to OPC UA server: {}", opcuaConfig.getEndpointUrl());
        } catch (Exception e) {
            log.error("Failed to connect to OPC UA server", e);
            throw new RuntimeException("OPC UA connection failed", e);
        }
    }

    /**
     * Lit une valeur depuis un node OPC UA
     */
    public SensorData readNode(OPCUAConfig.NodeConfig nodeConfig) {
        if (client == null) {
            log.warn("OPC UA client not initialized");
            return null;
        }

        try {
            NodeId nodeId = NodeId.parse(nodeConfig.getNodeId());
            DataValue dataValue = client.readValue(0.0, TimestampsToReturn.Both, nodeId).get();

            if (dataValue.getStatusCode().isGood()) {
                Variant variant = dataValue.getValue();
                Object value = variant.getValue();

                return SensorData.builder()
                        .timestamp(java.time.Instant.now())
                        .assetId(nodeConfig.getAssetId())
                        .sensorId(nodeConfig.getSensorId())
                        .value(convertToDouble(value))
                        .unit(nodeConfig.getUnit())
                        .quality(2) // Good
                        .sourceType("OPC_UA")
                        .sourceEndpoint(opcuaConfig.getEndpointUrl())
                        .build();
            } else {
                log.warn("Failed to read node {}: {}", nodeConfig.getNodeId(), dataValue.getStatusCode());
                return null;
            }
        } catch (Exception e) {
            log.error("Error reading OPC UA node: {}", nodeConfig.getNodeId(), e);
            return null;
        }
    }

    /**
     * Lit tous les nodes configurés
     */
    public List<SensorData> readAllNodes() {
        List<SensorData> dataList = new ArrayList<>();
        
        if (opcuaConfig.getNodes() == null) {
            return dataList;
        }

        for (OPCUAConfig.NodeConfig nodeConfig : opcuaConfig.getNodes()) {
            SensorData data = readNode(nodeConfig);
            if (data != null) {
                dataList.add(data);
            }
        }

        return dataList;
    }

    /**
     * Déconnecte du serveur OPC UA
     */
    public void disconnect() {
        if (client != null) {
            try {
                client.disconnect().get();
                log.info("Disconnected from OPC UA server");
            } catch (Exception e) {
                log.error("Error disconnecting from OPC UA server", e);
            }
        }
    }

    private Double convertToDouble(Object value) {
        if (value == null) {
            return null;
        }
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        try {
            return Double.parseDouble(value.toString());
        } catch (NumberFormatException e) {
            log.warn("Cannot convert value to double: {}", value);
            return null;
        }
    }
}

