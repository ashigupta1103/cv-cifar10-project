from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("cifar10_cnn_model.keras")
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_array = np.array(img.resize((32,32))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    

    pred = model.predict(img_array)
    predicted_class = class_names[np.argmax(pred)]
    
    return {"prediction": predicted_class}
