from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib  # or use pickle
import numpy as np
from typing import List
from tensorflow.keras.models import load_model

# Initialize the FastAPI app
app = FastAPI()

# Load the pre-trained model from Task 2
MODEL_PATH = '../result/lstm_model.h5'

# Load the model (Make sure the model is serialized with joblib/pickle)
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

# Define the input data model using Pydantic (for input validation)
class PredictionRequest(BaseModel):
    features: List[float]  # List of features for the model input


# Define the prediction response structure
class PredictionResponse(BaseModel):
    prediction: float


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionRequest):
    try:
        # Convert input data to a numpy array
        features = np.array(input_data.features).reshape(1, -1)

        # Make prediction using the loaded model
        prediction = model.predict(features)

        # Return the prediction as a response
        return PredictionResponse(prediction=prediction[0])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Run the API (This is for local testing)
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
