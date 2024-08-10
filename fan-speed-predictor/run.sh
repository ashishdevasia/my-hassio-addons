#!/usr/bin/with-contenv bashio

python3 -m uvicorn predict_fan_speed:app --host 0.0.0.0 --port 8000
