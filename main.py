from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from fetch import ReceiptPredictionModel  # Make sure this matches your model class in fetch.py

app = FastAPI()

# Static Files Configuration
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the request model
class PredictionRequest(BaseModel):
    month: int
    day_of_week: int

# Define the response model
class PredictionResponse(BaseModel):
    prediction: float

# Initialize your model (adjust as necessary)
model = ReceiptPredictionModel()
model.load_state_dict(torch.load('receipt_prediction_model.pth'))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    with open('static/index.html', 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Assuming your model takes month and day_of_week as inputs
        input_tensor = torch.tensor([[request.month, request.day_of_week]], dtype=torch.float32)
        prediction = model(input_tensor).item()  # Adjust according to your model's output
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add more endpoints as needed
