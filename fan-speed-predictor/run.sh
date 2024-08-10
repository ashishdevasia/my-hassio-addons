#!/usr/bin/with-contenv bashio

echo "Starting Fan Speed Predictor"
echo "Current directory: $(pwd)"
echo "Contents of current directory:"
ls -la

echo "Running Python script"
python3 -m uvicorn predict_fan_speed:app --host 0.0.0.0 --port 8000
