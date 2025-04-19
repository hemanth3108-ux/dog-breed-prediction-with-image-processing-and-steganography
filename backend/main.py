from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import List, Optional
import json
import os

class Prediction(BaseModel):
    breed: str
    probability: float

class PredictionResponse(BaseModel):
    predictions: List[Prediction]

class ErrorResponse(BaseModel):
    error: str
    details: str

app = FastAPI()

# Configure CORS - Allow all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False if allow_origins=["*"]
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize variables
model = None
class_names = [
    'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih-Tzu', 'Blenheim_spaniel', 'papillon',
    'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick',
    'black-and-tan_coonhound', 'Walker_hound', 'English_foxhound', 'redbone', 'borzoi', 'Irish_wolfhound',
    'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elkhound', 'otterhound', 'Saluki',
    'Scottish_deerhound', 'Weimaraner', 'Staffordshire_bullterrier', 'American_Staffordshire_terrier',
    'Bedlington_terrier', 'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier',
    'Norwich_terrier', 'Yorkshire_terrier', 'wire-haired_fox_terrier', 'Lakeland_terrier',
    'Sealyham_terrier', 'Airedale', 'cairn', 'Australian_terrier', 'Dandie_Dinmont', 'Boston_bull',
    'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier',
    'silky_terrier', 'soft-coated_wheaten_terrier', 'West_Highland_white_terrier', 'Lhasa',
    'flat-coated_retriever', 'curly-coated_retriever', 'golden_retriever', 'Labrador_retriever',
    'Chesapeake_Bay_retriever', 'German_short-haired_pointer', 'vizsla', 'English_setter', 'Irish_setter',
    'Gordon_setter', 'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel',
    'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael',
    'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog', 'collie',
    'Border_collie', 'Bouvier_des_Flandres', 'Rottweiler', 'German_shepherd', 'Doberman',
    'miniature_pinscher', 'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog', 'Appenzeller',
    'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog', 'Great_Dane',
    'Saint_Bernard', 'Eskimo_dog', 'malamute', 'Siberian_husky', 'dalmatian', 'affenpinscher',
    'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow',
    'keeshond', 'Brabancon_griffon', 'Pembroke', 'Cardigan', 'toy_poodle', 'miniature_poodle',
    'standard_poodle', 'Mexican_hairless'
]

# Load the model
try:
    print("Loading pre-trained ResNet50 model...")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.eval()
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"Error during initialization: {str(e)}")
    print(f"Error type: {type(e)}")
    raise

def preprocess_image(image):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    image_tensor = transform(image)
    print(f"Image tensor shape: {image_tensor.shape}")
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    print(f"Image tensor with batch: {image_tensor.shape}")
    
    return image_tensor

@app.get("/")
async def root():
    global model, class_names
    model_status = "loaded" if model is not None else "not loaded"
    num_classes = len(class_names) if class_names else 0
    return {
        "message": "Dog Breed Classification API",
        "model_status": model_status,
        "number_of_classes": num_classes,
        "sample_classes": class_names[:5] if class_names else None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    print(f"\n{'='*50}\nReceived prediction request\n{'='*50}")
    print(f"File: {file.filename}")
    
    if not model:
        error_msg = "Model is not loaded"
        print(f"Error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    
    try:
        # Read image
        print("Reading image file...")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        print(f"Image loaded successfully: size={image.size}, mode={image.mode}")
        
        # Preprocess image
        print("Preprocessing image...")
        processed_image = preprocess_image(image)
        print(f"Processed tensor shape: {processed_image.shape}, dtype={processed_image.dtype}")
        
        # Make prediction
        print("Making prediction...")
        with torch.no_grad():
            # Move tensor to same device as model
            # If you want to use GPU, also move the model to CUDA: model.to('cuda')
            processed_image = processed_image.to(next(model.parameters()).device)
            
            # Get model predictions
            predictions = model(processed_image)
            print(f"Raw predictions: shape={predictions.shape}, range=[{predictions.min().item():.2f}, {predictions.max().item():.2f}]")
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
            print(f"Probabilities sum: {probabilities.sum().item():.4f}")
        
        # Get dog breed predictions (ImageNet indices 151-268 are dog breeds)
        print("Extracting dog breed predictions...")
        dog_probs = probabilities[151:269]
        top_5_probs, top_5_relative_idx = torch.topk(dog_probs, k=5)
        top_5_idx = top_5_relative_idx + 151  # Convert back to ImageNet indices
        
        # Check if the image contains a dog by thresholding the highest probability
        max_prob = top_5_probs[0].item() if len(top_5_probs) > 0 else 0.0
        dog_threshold = 0.5  # You can adjust this threshold for stricter/looser detection
        if max_prob < dog_threshold:
            error_msg = "There is no dog in the image you have uploaded. Please upload a dog image."
            print(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Create predictions list
        top_5_predictions = []
        for i, (idx, prob) in enumerate(zip(top_5_idx.tolist(), top_5_probs.tolist())):
            # idx is the ImageNet index (151-268), class_names is ordered as 0-117 for dog breeds
            breed_idx = idx - 151
            breed = class_names[breed_idx] if 0 <= breed_idx < len(class_names) else f"Dog breed {idx}"
            probability = float(prob)
            print(f"  {i+1}. {breed}: {probability:.4f}")
            top_5_predictions.append(Prediction(breed=breed, probability=probability))
        
        print("Prediction complete!\n" + "="*50)
        
        # Return response
        response = PredictionResponse(predictions=top_5_predictions)
        print(f"Sending response: {response}")
        return response
    
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        print(f"\n{error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)