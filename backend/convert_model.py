import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import json
import os

class DogBreedClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load pretrained EfficientNet
        self.backbone = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        # Replace classifier
        self.backbone.classifier.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def convert_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tf_model_path = os.path.join(base_dir, 'model', 'dog_breed_model.h5')
    pt_model_path = os.path.join(base_dir, 'model', 'dog_breed_model.pth')
    class_names_path = os.path.join(base_dir, 'model', 'class_names.json')
    
    # Load class names to get number of classes
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    
    # Create PyTorch model
    model = DogBreedClassifier(num_classes)
    model.eval()
    
    # Load TensorFlow model weights
    tf_model = tf.keras.models.load_model(tf_model_path)
    
    # Convert weights
    # This is a simplified version - in practice, you'd need to map the weights correctly
    with torch.no_grad():
        for i, layer in enumerate(tf_model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()
                if len(weights) == 2:  # weights and bias
                    w, b = weights
                    model.backbone.classifier.fc.weight.copy_(torch.FloatTensor(w.T))
                    model.backbone.classifier.fc.bias.copy_(torch.FloatTensor(b))
    
    # Save PyTorch model
    torch.save(model, pt_model_path)
    print(f"Model saved to {pt_model_path}")

if __name__ == "__main__":
    convert_model()
