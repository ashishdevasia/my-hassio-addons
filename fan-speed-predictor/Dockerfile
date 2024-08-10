FROM python:3.9-slim-buster

ENV LANG C.UTF-8

# Install dependencies
RUN pip3 install --no-cache-dir tflite-runtime numpy joblib fastapi uvicorn

# Copy your prediction script and model
COPY predict_fan_speed.py /
COPY fan_speed_model.tflite /
COPY scaler.joblib /

# Copy the entry script and set permissions
COPY run.sh /
RUN chmod +x /run.sh && \
   sed -i 's/\r$//' /run.sh && \
   cat /run.sh

# Debug: List contents of root directory
RUN ls -la /

CMD ["/bin/bash", "/run.sh"]
