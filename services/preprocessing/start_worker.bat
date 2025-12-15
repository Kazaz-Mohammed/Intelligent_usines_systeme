@echo off
echo Starting Preprocessing Worker...
echo This will consume messages from Kafka topic: sensor-data
echo Press Ctrl+C to stop
echo.
cd /d "%~dp0"
python -m app.worker --mode streaming
pause

