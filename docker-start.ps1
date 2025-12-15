# Docker Start Script for Predictive Maintenance System (PowerShell)
# This script helps you start the entire system with Docker

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Predictive Maintenance System - Docker" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "What would you like to do?" -ForegroundColor Yellow
Write-Host "1) Start all services (build if needed)"
Write-Host "2) Start all services in background (detached)"
Write-Host "3) Start with optional tools (Kafka UI, pgAdmin)"
Write-Host "4) Stop all services"
Write-Host "5) Stop and remove all data (clean slate)"
Write-Host "6) View logs"
Write-Host "7) Check service status"
Write-Host ""
$choice = Read-Host "Enter your choice (1-7)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "🚀 Starting all services..." -ForegroundColor Green
        docker-compose up --build
    }
    "2" {
        Write-Host ""
        Write-Host "🚀 Starting all services in background..." -ForegroundColor Green
        docker-compose up -d --build
        Write-Host ""
        Write-Host "✅ Services are starting in the background" -ForegroundColor Green
        Write-Host "📊 Check status with: docker-compose ps" -ForegroundColor Cyan
        Write-Host "📝 View logs with: docker-compose logs -f" -ForegroundColor Cyan
    }
    "3" {
        Write-Host ""
        Write-Host "🚀 Starting all services with optional tools..." -ForegroundColor Green
        docker-compose --profile tools up -d --build
        Write-Host ""
        Write-Host "✅ Services are starting" -ForegroundColor Green
        Write-Host "🌐 Kafka UI: http://localhost:8080" -ForegroundColor Cyan
        Write-Host "🌐 pgAdmin: http://localhost:5050" -ForegroundColor Cyan
    }
    "4" {
        Write-Host ""
        Write-Host "🛑 Stopping all services..." -ForegroundColor Yellow
        docker-compose down
        Write-Host "✅ Services stopped" -ForegroundColor Green
    }
    "5" {
        Write-Host ""
        Write-Host "⚠️  WARNING: This will remove all data including databases!" -ForegroundColor Red
        $confirm = Read-Host "Are you sure? (yes/no)"
        if ($confirm -eq "yes") {
            docker-compose down -v
            Write-Host "✅ All services and data removed" -ForegroundColor Green
        } else {
            Write-Host "❌ Cancelled" -ForegroundColor Yellow
        }
    }
    "6" {
        Write-Host ""
        Write-Host "📝 Viewing logs (Ctrl+C to exit)..." -ForegroundColor Cyan
        docker-compose logs -f
    }
    "7" {
        Write-Host ""
        Write-Host "📊 Service Status:" -ForegroundColor Cyan
        docker-compose ps
    }
    default {
        Write-Host "❌ Invalid choice" -ForegroundColor Red
        exit 1
    }
}

