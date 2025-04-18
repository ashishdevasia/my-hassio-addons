import numpy as np
import tflite_runtime.interpreter as tflite
import joblib
from fastapi import FastAPI, HTTPException, Query
import sys
import pkg_resources
from datetime import datetime

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")

# Get TFLite Runtime version
try:
    tflite_version = pkg_resources.get_distribution("tflite-runtime").version
    print(f"TFLite Runtime version: {tflite_version}")
except pkg_resources.DistributionNotFound:
    print("TFLite Runtime version: Not found")

# Get scikit-learn version
try:
    sklearn_version = pkg_resources.get_distribution("scikit-learn").version
    print(f"scikit-learn version: {sklearn_version}")
except pkg_resources.DistributionNotFound:
    print("scikit-learn version: Not found")

class FanSpeedPredictor:
    def __init__(self, model_path, scaler_path):
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"Model loaded successfully. Input details: {self.input_details}, Output details: {self.output_details}")
            
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded successfully. Type: {type(self.scaler)}")
            
            print("FanSpeedPredictor initialized successfully")
        except Exception as e:
            print(f"Error initializing FanSpeedPredictor: {str(e)}")
            raise

    def predict(self, temperature, humidity, heart_rate):
        try:
            input_data = np.array([[temperature, humidity, heart_rate]], dtype=np.float32)
            scaled_input = self.scaler.transform(input_data)
            self.interpreter.set_tensor(self.input_details[0]['index'], scaled_input)
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
            return float(prediction[0][0])
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise

app = FastAPI()

try:
    predictor = FanSpeedPredictor('/fan_speed_model.tflite', '/scaler.joblib')
    print("Predictor created successfully")
except Exception as e:
    print(f"Failed to create predictor: {str(e)}")
    raise

@app.get("/predict")
async def predict_fan_speed(
    temperature: float = Query(..., description="Temperature value"),
    humidity: float = Query(..., description="Humidity value"),
    heart_rate: float = Query(..., description="Heart rate value")
):
    try:
        fan_speed = predictor.predict(temperature, humidity, heart_rate)
        print(f'{datetime.now()}: Predicted Fan Speed: {round(fan_speed, 2)}')
        return {
            "predicted_fan_speed": round(fan_speed, 2),
            "temperature": temperature,
            "humidity": humidity,
            "heart_rate": heart_rate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

print("FastAPI app created successfully")
