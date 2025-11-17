# backend/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
model = tf.keras.models.load_model('ortho_model.h5')
classes = ['Clase I', 'Clase II', 'Clase III']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer y preprocesar imagen
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predicción
    predictions = model.predict(image_array)
    predicted_class = classes[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    return {
        "classification": predicted_class,
        "confidence": confidence,
        "timestamp": "2024-01-01T00:00:00Z"  # Usar datetime real
    }

@app.get("/")
async def root():
    return {"message": "Ortho ML API running"}