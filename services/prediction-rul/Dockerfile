FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements.txt
COPY requirements.txt .

# Installer les dépendances Python en plusieurs étapes pour éviter de perdre le progrès
# Strategy: Install in batches - small packages first, then large ones separately
# This way, if a large package fails, we don't lose progress on smaller ones
# Skip pip upgrade to avoid timeout issues - pip 24.0+ from base image is sufficient

# Helper function to install package with retry
RUN printf 'install_pkg() {\n\
    local pkg="$1";\n\
    local max=12;\n\
    for i in $(seq 1 $max); do\n\
        echo "[$i/$max] Installing $pkg...";\n\
        if pip install --no-cache-dir --timeout=600 --default-timeout=600 "$pkg"; then\n\
            echo "✓ Success: $pkg";\n\
            return 0;\n\
        fi;\n\
        delay=$((i * 20));\n\
        echo "✗ Failed, retry in ${delay}s...";\n\
        [ $i -lt $max ] && sleep $delay || return 1;\n\
    done;\n\
}\n' > /tmp/install_pkg.sh && chmod +x /tmp/install_pkg.sh

# Stage 1: Small essential packages (FastAPI, Kafka, etc.)
RUN . /tmp/install_pkg.sh && \
    install_pkg "fastapi==0.104.1" && \
    install_pkg "uvicorn[standard]==0.24.0" && \
    install_pkg "pydantic==2.5.0" && \
    install_pkg "pydantic-settings==2.1.0" && \
    install_pkg "confluent-kafka==2.3.0" && \
    install_pkg "python-json-logger==2.0.7" && \
    install_pkg "psycopg2-binary==2.9.9"

# Stage 2: Medium data processing packages
RUN . /tmp/install_pkg.sh && \
    install_pkg "numpy==1.26.2" && \
    install_pkg "pandas==2.1.3" && \
    install_pkg "scipy==1.11.4" && \
    install_pkg "scikit-learn==1.3.2"

# Stage 3: Large ML packages (install separately with more retries)
RUN . /tmp/install_pkg.sh && \
    install_pkg "xgboost==2.0.3" && \
    install_pkg "mlflow==2.8.1"

# Stage 4: Very large Deep Learning packages (install one at a time)
RUN . /tmp/install_pkg.sh && \
    install_pkg "torch==2.1.0" && \
    install_pkg "torchvision==0.16.0"

# Stage 5: Development packages (non-critical, continue on failure)
RUN pip install --no-cache-dir --timeout=600 --default-timeout=600 \
    pytest==7.4.3 pytest-asyncio==0.21.1 pytest-cov==4.1.0 pytest-mock==3.12.0 \
    black==23.11.0 flake8==6.1.0 mypy==1.7.1 || \
    echo "⚠ Warning: Some dev packages failed, continuing..."

# Cleanup
RUN rm -f /tmp/install_pkg.sh

# Copier le code de l'application
COPY app ./app
COPY start_worker.py ./start_worker.py
COPY models ./models

# Exposer le port
EXPOSE 8085

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8085/health')" || exit 1

# Commande de démarrage
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8085"]

