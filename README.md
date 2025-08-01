# 🦟 Mosquito Detection AI

A deep learning-powered web application that can detect mosquitoes in images using computer vision. Built with PyTorch and Flask, featuring a modern drag-and-drop interface.

![Mosquito Detection Demo](https://img.shields.io/badge/AI-Computer%20Vision-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red) ![Flask](https://img.shields.io/badge/Flask-2.3.3-lightgrey)

## ✨ Features

- **🔬 High Accuracy Detection**: Uses EfficientNet-B0 architecture with transfer learning
- **🖱️ Drag & Drop Interface**: Modern, intuitive web interface with Apple San Francisco font
- **⚡ Real-time Predictions**: Instant mosquito detection with confidence scores
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices
- **🎯 Binary Classification**: Accurately distinguishes between mosquito and non-mosquito images

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kaushalpemmaraju/Mosquito-Detection.git
   cd Mosquito-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   # Open the training notebook
   jupyter notebook Model_training.ipynb
   # Run all cells to train and save the model
   ```

4. **Run the web application**
   ```bash
   python app.py
   ```

5. **Open your browser** and navigate to `http://localhost:5001`

## 📊 Model Training

The model is trained using the Jupyter notebook `Model_training.ipynb`. Here's what it includes:

### Dataset Structure
```
data/
├── Mosquito/          # Mosquito images
│   ├── Mosquito_1.jpg
│   ├── Mosquito_2.jpg
│   └── ...
└── Not_mosquito/      # Non-mosquito images
    ├── Image_1.jpg
    ├── Image_2.jpg
    └── ...
```

### Training Process
1. **Data Preprocessing**: Images are resized to 224x224 and normalized using ImageNet statistics
2. **Data Augmentation**: Random horizontal flips and resizing for better generalization
3. **Transfer Learning**: Uses pre-trained EfficientNet-B0 as the base model
4. **Fine-tuning**: Only the final classification layer is trained (feature extraction approach)
5. **Train/Test Split**: 80% training, 20% testing

### Key Training Parameters
- **Model**: EfficientNet-B0 (pre-trained on ImageNet)
- **Optimizer**: SGD with momentum (0.9) and learning rate (0.001)
- **Loss Function**: CrossEntropyLoss
- **Scheduler**: StepLR (step_size=5, gamma=0.1)
- **Epochs**: 20
- **Batch Size**: 32

## 🏗️ Project Structure

```
Mosquito-Detection/
├── app.py                    # Flask web application
├── Model_training.ipynb      # Jupyter notebook for training
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── LICENSE                  # MIT License
├── data/                    # Training dataset
│   ├── Mosquito/           # Mosquito images
│   └── Not_mosquito/       # Non-mosquito images
├── templates/               # HTML templates
│   └── index.html          # Web interface
└── model/                   # Saved model files (generated after training)
    ├── mosquito_detection_model.pth  # Trained model weights
    └── class_names.txt      # Class labels
```

**Note**: The `model/` contents are generated after training and may not be included in the repository initially.

## 🔧 Technical Details

### Model Architecture
- **Base Model**: EfficientNet-B0 (1,280 features)
- **Classification Head**: Linear layer (1,280 → 2 classes)
- **Device Support**: Apple Silicon (MPS), CUDA, and CPU

### Web Application
- **Backend**: Flask 2.3.3
- **Frontend**: HTML5, CSS3, JavaScript
- **Image Processing**: PIL (Pillow)
- **Real-time Predictions**: AJAX requests

### Performance Optimizations
- **Transfer Learning**: Freezes base model parameters for faster training
- **MPS Support**: Optimized for Apple Silicon GPUs
- **Efficient Inference**: Model runs in evaluation mode for predictions

## 📈 Usage Examples

### Training a New Model
```python
# Run the training notebook
jupyter notebook Model_training.ipynb

# Or train programmatically
python -c "
from Model_training import train_model
model = train_model(epochs=20, batch_size=32)
"
```

### Making Predictions
```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = torch.load('model/mosquito_detection_model.pth')
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Make prediction
image = Image.open('test_image.jpg')
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, 1)
```

## 🎯 Model Performance

The model achieves high accuracy on the validation set:
- **Architecture**: EfficientNet-B0 with transfer learning
- **Training Strategy**: Feature extraction (frozen backbone)
- **Validation Split**: 20% of total dataset
- **Real-time Inference**: < 1 second per image

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Contribution
- 🆕 Additional model architectures (ResNet, Vision Transformer)
- 📊 Model performance metrics and visualizations
- 🎨 UI/UX improvements
- 📱 Mobile app development
- 🔄 Data augmentation techniques
- ⚡ Performance optimizations

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **EfficientNet Authors** for the efficient CNN architecture
- **ImageNet** for providing pre-trained weights
- **Flask Community** for the lightweight web framework
- **Kaggle Dataset**: Mosquito images from [Insects Dataset](https://www.kaggle.com/datasets/ismail703/insects/data) by ismail703


## 📞 Contact

**kaushalpemmaraju** - [GitHub Profile](https://github.com/kaushalpemmaraju)

Project Link: [https://github.com/kaushalpemmaraju/Mosquito-Detection](https://github.com/kaushalpemmaraju/Mosquito-Detection)

---

⭐ **Star this repository if you found it helpful!** ⭐
A user-friendly web app that identifies mosquito species from uploaded images, helping users quickly recognize and learn about different types of mosquitoes for better awareness and prevention.
