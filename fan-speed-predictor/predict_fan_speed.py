import numpy as np
import tflite_runtime.interpreter as tflite
import joblib
from fastapi import FastAPI

class FanSpeedPredictor:
   def __init__(self, model_path, scaler_path):
       self.interpreter = tflite.Interpreter(model_path=model_path)
       self.interpreter.allocate_tensors()
       self.input_details = self.interpreter.get_input_details()
       self.output_details = self.interpreter.get_output_details()
       self.scaler = joblib.load(scaler_path)

   def predict(self, temperature, humidity, heart_rate):
       input_data = np.array([[temperature, humidity, heart_rate]], dtype=np.float32)
       scaled_input = self.scaler.transform(input_data)
       self.interpreter.set_tensor(self.input_details[0]['index'], scaled_input)
       self.interpreter.invoke()
       prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
       return max(min(round(float(prediction[0][0])), 5), 0)

predictor = FanSpeedPredictor('/fan_speed_model.tflite', '/scaler.joblib')

app = FastAPI()

@app.post("/predict")
async def predict_fan_speed(temperature: float, humidity: float, heart_rate: float):
   fan_speed = predictor.predict(temperature, humidity, heart_rate)
   return {"predicted_fan_speed": round(fan_speed, 2)}
