import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
from flask import Flask, render_template, request, jsonify
import io
import base64

app = Flask(__name__)

# Load the trained model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def load_model():
    # Load the model architecture
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    
    # Modify for 2 classes
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    
    # Load the trained weights
    model_path = 'model/mosquito_detection_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print("Model file not found! Make sure to train and save the model first.")
    
    model.to(device)
    model.eval()
    return model

# Load class names
def load_class_names():
    class_names_path = 'model/class_names.txt'
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    else:
        return ['Mosquito', 'Not_mosquito']  # Default class names

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load model and class names
model = load_model()
class_names = load_class_names()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Read and process the image
        image = Image.open(file.stream)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            _, predicted = torch.max(outputs, 1)
            
        # Get prediction results
        predicted_class = class_names[predicted.item()]
        confidence = probabilities[predicted.item()].item() * 100
        
        # Get probabilities for both classes
        class_probabilities = {}
        for i, class_name in enumerate(class_names):
            class_probabilities[class_name] = float(probabilities[i].item() * 100)
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'probabilities': class_probabilities
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
