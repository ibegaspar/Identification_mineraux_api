from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import joblib
import keras
import os
import io
from PIL import Image
import uvicorn
import tensorflow as tf
import pandas as pd

# Afficher la version de TensorFlow
print(f"🔧 TensorFlow version: {tf.__version__}")

app = FastAPI(
    title="API Prédiction Minéraux",
    description="API pour identifier les minéraux à partir d'images et propriétés physiques",
    version="1.0.0"
)

# Configuration CORS pour permettre les requêtes depuis l'application mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement des modèles et outils
try:
    model = keras.models.load_model('model_final_durete_densiter.h5')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("✅ Modèles chargés avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement des modèles : {e}")
    model = None
    scaler = None
    label_encoder = None

IMG_SIZE = (380, 380)

def preprocess_image(image_bytes):
    """Prétraitement de l'image"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return img

@app.get("/")
async def root():
    """Endpoint de test"""
    return {"message": "API Prédiction Minéraux - Team RobotMali", "status": "active"}

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "available_endpoints": ["/predict", "/health", "/"]
    }

@app.post("/predict")
async def predict_mineral(
    image: UploadFile = File(...),
    durete: float = Form(...),
    densite: float = Form(...)
):
    """
    Prédiction de minéral à partir d'une image et des propriétés physiques
    
    Args:
        image: Image du minéral (JPG, PNG)
        durete: Dureté sur l'échelle de Mohs (0-10)
        densite: Densité en g/cm³ (0-20)
    
    Returns:
        Prédiction du minéral avec confiance
    """
    
    # Vérification des modèles
    if model is None or scaler is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Modèles non disponibles")
    
    # Vérification des paramètres
    if durete < 0 or durete > 10:
        raise HTTPException(status_code=400, detail="Dureté doit être entre 0 et 10")
    if densite < 0 or densite > 20:
        raise HTTPException(status_code=400, detail="Densité doit être entre 0 et 20")
    
    try:
        # Lecture de l'image
        image_bytes = await image.read()
        
        # Vérification du format
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Fichier doit être une image")
        
        # Prétraitement
        img = preprocess_image(image_bytes)
        X_img = np.expand_dims(img, axis=0)
        # Utiliser un DataFrame avec noms de colonnes pour scaler
        X_tab = scaler.transform(pd.DataFrame([[durete, densite]], columns=pd.Index(['durete', 'densite'])))
        # Prédiction
        prediction = model.predict([X_img, X_tab], verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_mineral = label_encoder.inverse_transform([predicted_index])[0]
        confidence = float(np.max(prediction[0]) * 100)
        
        return {
            "success": True,
            "predicted_mineral": predicted_mineral,
            "confidence": confidence,
            "input_data": {
                "durete": durete,
                "densite": densite,
                "image_size": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}"
            },
            "model_info": {
                "model_loaded": True,
                "prediction_time": "real-time"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

@app.post("/predict_simple")
async def predict_mineral_simple(
    durete: float = Form(...),
    densite: float = Form(...)
):
    """
    Prédiction simple basée uniquement sur les propriétés physiques
    (sans image, pour les cas où l'image n'est pas disponible)
    """
    
    if durete < 0 or durete > 10:
        raise HTTPException(status_code=400, detail="Dureté doit être entre 0 et 10")
    if densite < 0 or densite > 20:
        raise HTTPException(status_code=400, detail="Densité doit être entre 0 et 20")
    
    try:
        # Liste des minéraux avec leurs plages de dureté et densité
        minerals = [
            {"name": "azurite", "durete_min": 3.5, "durete_max": 4.0, "densite_min": 3.77, "densite_max": 3.77},
            {"name": "copper", "durete_min": 2.5, "durete_max": 3.0, "densite_min": 8.9, "densite_max": 8.9},
            {"name": "galena", "durete_min": 2.5, "durete_max": 2.5, "densite_min": 7.6, "densite_max": 7.6},
            {"name": "gold", "durete_min": 2.5, "durete_max": 3.0, "densite_min": 15.0, "densite_max": 19.3},
            {"name": "hematite", "durete_min": 5.0, "durete_max": 6.0, "densite_min": 5.26, "densite_max": 5.26},
            {"name": "limonite", "durete_min": 4.0, "durete_max": 5.5, "densite_min": 2.7, "densite_max": 4.3},
            {"name": "magnetite", "durete_min": 5.5, "durete_max": 6.5, "densite_min": 5.175, "densite_max": 5.175},
            {"name": "opal", "durete_min": 5.5, "durete_max": 6.5, "densite_min": 1.9, "densite_max": 2.3},
            {"name": "pyrit", "durete_min": 6.0, "durete_max": 6.5, "densite_min": 4.8, "densite_max": 5.0},
            {"name": "quartz", "durete_min": 7.0, "durete_max": 7.0, "densite_min": 2.6, "densite_max": 2.6},
            {"name": "saphir", "durete_min": 9.0, "durete_max": 9.0, "densite_min": 3.98, "densite_max": 4.06},
            {"name": "schwefel", "durete_min": 1.5, "durete_max": 2.5, "densite_min": 2.07, "densite_max": 2.07},
        ]
        predicted_mineral = "Minéral commun"
        confidence = 70.0
        for mineral in minerals:
            if mineral["durete_min"] <= durete <= mineral["durete_max"] and mineral["densite_min"] <= densite <= mineral["densite_max"]:
                predicted_mineral = mineral["name"]
                confidence = 90.0
                break
        
        return {
            "success": True,
            "predicted_mineral": predicted_mineral,
            "confidence": confidence,
            "method": "physical_properties_only",
            "input_data": {
                "durete": durete,
                "densite": densite
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 