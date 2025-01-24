FROM python:3.9-slim-buster

ENV LANG C.UTF-8

# Install dependencies
RUN pip3 install --no-cache-dir numpy==2.0.2 joblib==1.2.0 scikit-learn==1.0.2 fastapi==0.95.0 uvicorn==0.21.1
RUN pip3 install --no-cache-dir tflite-runtime==2.14.0

# Copy your prediction script and model
COPY predict_fan_speed.py /
COPY fan_speed_model.tflite /
COPY scaler.joblib /

# Copy the entry script and set permissions
COPY run.sh /
RUN chmod +x /run.sh && \
    sed -i 's/\r$//' /run.sh

# Debug: List contents of root directory
RUN ls -la /

# Print Python and pip versions
RUN python --version && pip --version

CMD ["/bin/bash", "/run.sh"]
