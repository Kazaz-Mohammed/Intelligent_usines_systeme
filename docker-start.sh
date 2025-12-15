#!/bin/bash

# Docker Start Script for Predictive Maintenance System
# This script helps you start the entire system with Docker

set -e

echo "=========================================="
echo "Predictive Maintenance System - Docker"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Ask user what to do
echo "What would you like to do?"
echo "1) Start all services (build if needed)"
echo "2) Start all services in background (detached)"
echo "3) Start with optional tools (Kafka UI, pgAdmin)"
echo "4) Stop all services"
echo "5) Stop and remove all data (clean slate)"
echo "6) View logs"
echo "7) Check service status"
echo ""
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting all services..."
        docker-compose up --build
        ;;
    2)
        echo ""
        echo "🚀 Starting all services in background..."
        docker-compose up -d --build
        echo ""
        echo "✅ Services are starting in the background"
        echo "📊 Check status with: docker-compose ps"
        echo "📝 View logs with: docker-compose logs -f"
        ;;
    3)
        echo ""
        echo "🚀 Starting all services with optional tools..."
        docker-compose --profile tools up -d --build
        echo ""
        echo "✅ Services are starting"
        echo "🌐 Kafka UI: http://localhost:8080"
        echo "🌐 pgAdmin: http://localhost:5050"
        ;;
    4)
        echo ""
        echo "🛑 Stopping all services..."
        docker-compose down
        echo "✅ Services stopped"
        ;;
    5)
        echo ""
        echo "⚠️  WARNING: This will remove all data including databases!"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            docker-compose down -v
            echo "✅ All services and data removed"
        else
            echo "❌ Cancelled"
        fi
        ;;
    6)
        echo ""
        echo "📝 Viewing logs (Ctrl+C to exit)..."
        docker-compose logs -f
        ;;
    7)
        echo ""
        echo "📊 Service Status:"
        docker-compose ps
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

