import torch
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import numpy as np

from src import mnist_loader, network

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

training_data, validation_data, test_data = mnist_loader.load_data()
net = network.Network([784, 30, 10])

model_path = "./trained_data/mnist_net.pth"
net.load_model(model_path)

class ImageData(BaseModel):
    image: list[float]

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/network/predict", tags=["network"])
async def predict(image: ImageData):
    image_array = np.array(image.image).reshape(784, 1)
    tensor_img = torch.tensor(image_array.ravel(), dtype=torch.float32)
    prediction = torch.argmax(net.forward(tensor_img)).item()
    return {"result": prediction}