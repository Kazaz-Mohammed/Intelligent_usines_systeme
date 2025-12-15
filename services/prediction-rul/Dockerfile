FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements.txt
COPY requirements.txt .

# Installer les dépendances Python avec timeout augmenté et retries
# Skip pip upgrade to avoid timeout issues - pip 24.0+ from base image is sufficient
RUN set -e; \
    for i in 1 2 3 4 5; do \
        if pip install --no-cache-dir --timeout=300 --default-timeout=300 -r requirements.txt; then \
            echo "Dependencies installed successfully on attempt $i"; \
            break; \
        else \
            echo "Install attempt $i failed, retrying..." && sleep 15; \
            if [ $i -eq 5 ]; then \
                echo "Failed to install dependencies after 5 attempts"; \
                exit 1; \
            fi; \
        fi; \
    done

# Copier le code de l'application
COPY app ./app

# Exposer le port
EXPOSE 8085

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8085/health')" || exit 1

# Commande de démarrage
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8085"]

