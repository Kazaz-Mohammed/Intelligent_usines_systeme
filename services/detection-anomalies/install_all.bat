@echo off
echo Installing all dependencies for detection-anomalies service...
echo This may take 10-15 minutes, especially for PyTorch...

cd /d "%~dp0"

REM Core FastAPI packages
python -m pip install fastapi uvicorn[standard] pydantic pydantic-settings

REM Data processing
python -m pip install pandas numpy scipy

REM Kafka and schema registry dependencies
python -m pip install confluent-kafka cachetools fastavro jsonschema referencing authlib cryptography attrs certifi httpx

REM ML packages
python -m pip install pyod scikit-learn

REM PyTorch (CPU version - faster to install)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

REM MLflow and database
python -m pip install mlflow psycopg2-binary python-json-logger

REM Testing and dev tools
python -m pip install pytest pytest-asyncio pytest-cov pytest-mock black flake8 mypy

echo.
echo Installation complete!
echo You can now run: python -m uvicorn app.main:app --host 0.0.0.0 --port 8084
pause

