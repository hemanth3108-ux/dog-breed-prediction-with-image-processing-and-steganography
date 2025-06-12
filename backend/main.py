from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import List, Optional
import json
import os
from stegano import lsb
import uuid
import base64
from image_processing_api import router as image_processing_router

class Prediction(BaseModel):
    breed: str
    probability: float

class PredictionResponse(BaseModel):
    predictions: List[Prediction]

class ErrorResponse(BaseModel):
    error: str
    details: str

class SteganographyRequest(BaseModel):
    message: str
    save_path: str

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

# Add this mapping at the top of your file (after imports)
imagenet_dog_classes = {
    151: "Chihuahua",
    152: "Japanese Spaniel",
    153: "Maltese Dog",
    154: "Pekinese",
    155: "Shih-Tzu",
    156: "Blenheim Spaniel",
    157: "Papillon",
    158: "Toy Terrier",
    159: "Rhodesian Ridgeback",
    160: "Afghan Hound",
    161: "Basset",
    162: "Beagle",
    163: "Bloodhound",
    164: "Bluetick",
    165: "Black-and-tan Coonhound",
    166: "Walker Hound",
    167: "English Foxhound",
    168: "Redbone",
    169: "Borzoi",
    170: "Irish Wolfhound",
    171: "Italian Greyhound",
    172: "Whippet",
    173: "Ibizan Hound",
    174: "Norwegian Elkhound",
    175: "Otterhound",
    176: "Saluki",
    177: "Scottish Deerhound",
    178: "Weimaraner",
    179: "Staffordshire Bullterrier",
    180: "American Staffordshire Terrier",
    181: "Bedlington Terrier",
    182: "Border Terrier",
    183: "Kerry Blue Terrier",
    184: "Irish Terrier",
    185: "Norfolk Terrier",
    186: "Norwich Terrier",
    187: "Yorkshire Terrier",
    188: "Wire-haired Fox Terrier",
    189: "Lakeland Terrier",
    190: "Sealyham Terrier",
    191: "Airedale",
    192: "Cairn",
    193: "Australian Terrier",
    194: "Dandie Dinmont",
    195: "Boston Bull",
    196: "Miniature Schnauzer",
    197: "Giant Schnauzer",
    198: "Standard Schnauzer",
    199: "Scotch Terrier",
    200: "Tibetan Terrier",
    201: "Silky Terrier",
    202: "Soft-coated Wheaten Terrier",
    203: "West Highland White Terrier",
    204: "Lhasa",
    205: "Flat-coated Retriever",
    206: "Curly-coated Retriever",
    207: "Golden Retriever",
    208: "Labrador Retriever",
    209: "Chesapeake Bay Retriever",
    210: "German Short-haired Pointer",
    211: "Vizsla",
    212: "English Setter",
    213: "Irish Setter",
    214: "Gordon Setter",
    215: "Brittany Spaniel",
    216: "Clumber",
    217: "English Springer",
    218: "Welsh Springer Spaniel",
    219: "Cocker Spaniel",
    220: "Sussex Spaniel",
    221: "Irish Water Spaniel",
    222: "Kuvasz",
    223: "Schipperke",
    224: "Groenendael",
    225: "Malinois",
    226: "Briard",
    227: "Kelpie",
    228: "Komondor",
    229: "Old English Sheepdog",
    230: "Shetland Sheepdog",
    231: "Collie",
    232: "Border Collie",
    233: "Bouvier des Flandres",
    234: "Rottweiler",
    235: "German Shepherd",
    236: "Doberman",
    237: "Miniature Pinscher",
    238: "Greater Swiss Mountain Dog",
    239: "Bernese Mountain Dog",
    240: "Appenzeller",
    241: "EntleBucher",
    242: "Boxer",
    243: "Bull Mastiff",
    244: "Tibetan Mastiff",
    245: "French Bulldog",
    246: "Great Dane",
    247: "Saint Bernard",
    248: "Eskimo Dog",
    249: "Malamute",
    250: "Siberian Husky"
    # ... (add more if needed)
}

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

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    use_steganography: bool = Form(False),
    message: Optional[str] = Form(None),
    save_path: Optional[str] = Form(None)
):
    try:
        # Read the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # If steganography is requested
        if use_steganography and message:
            if not save_path:
                save_path = f"steg_{uuid.uuid4()}.png"
            
            # Save the original image temporarily
            temp_path = f"temp_{uuid.uuid4()}.png"
            image.save(temp_path)
            
            # Hide the message
            secret = lsb.hide(temp_path, message)
            secret.save(save_path)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            # Return the path to the steganographed image
            return FileResponse(
                save_path,
                media_type="image/png",
                headers={"X-Steganography": "true"}
            )
        
        # Preprocess the image
        image_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            
        # Prepare response
        predictions = []
        for i in range(top5_prob.size(0)):
            idx = top5_catid[i].item()
            breed = imagenet_dog_classes.get(idx, f"Class {idx}")
            predictions.append({
                "breed": breed,
                "probability": float(top5_prob[i])
            })
        
        return {"predictions": predictions}
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-message")
async def extract_message(file: UploadFile = File(...)):
    try:
        # Read the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Save temporarily
        temp_path = f"temp_{uuid.uuid4()}.png"
        image.save(temp_path)
        
        # Extract the message
        message = lsb.reveal(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return {"message": message}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/encode")
async def encode_image(
    file: UploadFile = File(...),
    message: str = Form(...),
    save_path: Optional[str] = Form(None),
    save_to_server: bool = Form(False)
):
    try:
        # Read the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Generate a unique filename if save_path not provided
        if not save_path:
            save_path = f"encoded_{uuid.uuid4()}.png"
        
        # Save the original image temporarily
        temp_path = f"temp_{uuid.uuid4()}.png"
        image.save(temp_path)
        
        # Encode the message
        secret = lsb.hide(temp_path, message)
        secret.save(save_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        # Read the encoded image and convert to base64
        with open(save_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        server_path = None
        if not save_to_server:
            # Clean up the saved file if not saving to server
            os.remove(save_path)
        else:
            server_path = os.path.abspath(save_path)
        
        return {
            "success": True,
            "encoded_image": encoded_image,
            "message": "Message encoded successfully",
            "server_path": server_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decode")
async def decode_image(file: UploadFile = File(...)):
    try:
        # Read the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Save temporarily
        temp_path = f"temp_{uuid.uuid4()}.png"
        image.save(temp_path)
        
        # Decode the message
        try:
            message = lsb.reveal(temp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail="No hidden message found in the image")
        
        # Clean up
        os.remove(temp_path)
        
        return {
            "success": True,
            "message": message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(image_processing_router)