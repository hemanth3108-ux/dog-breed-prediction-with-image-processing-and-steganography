# Dog Breed Classification App

This is a full-stack application that uses machine learning to classify dog breeds from images. The application consists of a Python backend using FastAPI and TensorFlow, and a React frontend with TypeScript.

## Project Structure

```
.
├── backend/
│   ├── main.py           # FastAPI server
│   ├── train_model.py    # Model training script
│   └── model/           # Directory for saved model files
├── frontend/
│   ├── src/
│   │   └── App.tsx      # Main React component
│   └── package.json     # Frontend dependencies
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn
- The dog breed dataset in the correct structure

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   cd backend
   python train_model.py
   ```

3. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

## Running the Application

1. Start the backend server:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. Start the frontend development server:
   ```bash
   cd frontend
   npm start
   ```

3. Open your browser and navigate to `http://localhost:3000`

## Usage

1. The web interface will show a drag-and-drop area for uploading dog images
2. Upload an image of a dog
3. The application will display the top 5 predicted breeds with confidence scores

## Model Details

The model uses EfficientNetB0 as the base model with transfer learning. It's trained on the dog breed identification dataset with the following characteristics:
- Input size: 224x224 pixels
- Number of classes: Based on the dataset
- Training augmentation: rotation, shift, shear, zoom, and horizontal flip

## API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Accepts an image file and returns breed predictions

## Notes

- The model file is not included in the repository due to size constraints
- You need to train the model before using the application
- Make sure the dataset is properly organized in the required directory structure 