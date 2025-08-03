# 🧠 CNN Animal Classifier using MobileNetV2 Transfer Learning

This project implements a **deep learning image classification model** using **MobileNetV2** and **transfer learning** with the **TensorFlow/Keras** framework. The model classifies animal images into three categories: **cats**, **dogs**, and **snakes** with high accuracy using a pretrained convolutional neural network.

## 🚀 Project Features

- ✅ **Transfer Learning** with MobileNetV2 (ImageNet pretrained weights)
- ✅ **Frozen Base Layers** for efficient training with limited data
- ✅ **Data Augmentation** (rotation, zoom, flip) for better generalization
- ✅ **Smart Callbacks** (EarlyStopping, ReduceLROnPlateau)
- ✅ **Comprehensive Evaluation** with confusion matrix and metrics
- ✅ **Training Visualization** (loss/accuracy curves, confidence analysis)
- ✅ **Model Persistence** with h5 formats (.h5)
- ✅ **Production-Ready** code structure following best practices

---

## 📂 Project Structure
```
Image-Classifier-MobileNetV2/
├── Animals/                    # Dataset folder
│   ├── cats/                  # Cat images
│   ├── dogs/                  # Dog images
│   └── snakes/                # Snake images
│
├── notebooks/
│   └── animal_classifier.ipynb    # Main training notebook
│
├── saved_models/
│   ├── animal_classifier_model.h5         # Keras model (.h5)
│
├── evaluation_results/
│   ├── predictions.npy               # Model predictions
│   ├── true_classes.npy             # Ground truth labels
│   └── predicted_classes.npy        # Predicted class indices
│
├── README.md                    # Project documentation
└── requirements.txt            # Python dependencies
```

---

## 🧰 Technologies Used

- **Python 3.8+**
- **TensorFlow 2.x / Keras** - Deep learning framework
- **MobileNetV2** - Lightweight CNN architecture
- **scikit-learn** - Evaluation metrics
- **Matplotlib / Seaborn** - Data visualization
- **NumPy** - Numerical computing
- **Jupyter Notebook** - Development environment

---

## 📊 Dataset Requirements

The dataset should be organized in the following folder structure:
```
Animals/
├── cats/       # 1000 images of cats
├── dogs/       # 1000 images of dogs
└── snakes/     # 1000 images of snakes
```

**Dataset Statistics:**
- **Total Images**: ~3000 images
- **Classes**: 3 (cats, dogs, snakes)
- **Split**: 80% training, 20% validation
- **Image Size**: 224×224 pixels (resized automatically)
- **Format**: JPG, PNG supported

---

## 🏗️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/DPHeshanRanasinghe/Image-Classifier-MobileNetV2.git
cd Image-Classifier-MobileNetV2
```

### 2. Set Up Environment
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Prepare Your Dataset
- Organize your animal images in the `Animals/` folder structure shown above
- Ensure each class folder contains at least 500+ images for good performance

### 4. Run the Training
```bash
# Open Jupyter Notebook
jupyter notebook

# Or run in Google Colab
# Upload the notebook and dataset to Colab
```

### 5. Execute the Notebook
Run the cells step-by-step:
1. **Data Loading & Exploration** - Visualize dataset distribution
2. **Data Preprocessing** - Augmentation and generators
3. **Model Creation** - MobileNetV2 with custom head
4. **Training** - With callbacks and monitoring
5. **Evaluation** - Comprehensive performance analysis
6. **Model Saving** - Multiple formats for deployment

---

## 🎯 Model Architecture

```
Input (224×224×3)
        ↓
MobileNetV2 Base (Frozen)
   - ImageNet Weights
   - 154 Layers
   - 2.3M Parameters
        ↓
GlobalAveragePooling2D
        ↓
Dense(128, ReLU)
        ↓
Dropout(0.5)
        ↓
Dense(3, Softmax)
        ↓
Output [cats, dogs, snakes]
```

**Training Configuration:**
- **Optimizer**: Adam (lr=0.0001)
- **Loss**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 15 (with EarlyStopping)
- **Data Augmentation**: Rotation, zoom, flip, shift

---

## 📈 Results & Performance

### 🎯 Model Performance
- **Final Validation Accuracy**: **~95-98%**
- **Training Time**: ~10-15 minutes (depending on hardware)
- **Model Size**: ~9MB (.h5 format)
- **Inference Speed**: ~50ms per image

### 📊 Evaluation Metrics
```
Classification Report:
              precision    recall  f1-score   support

        cats     0.9930    0.9500    0.9710       300
        dogs     0.9553    0.9967    0.9755       300
      snakes     0.9933    0.9933    0.9933       300

    accuracy                         0.9800       900
   macro avg     0.9805    0.9800    0.9800       900
weighted avg     0.9805    0.9800    0.9800       900
```

### 🎨 Key Visualizations
- **Training Curves**: Loss and accuracy progression
- **Confusion Matrix**: Classification performance per class
- **Confidence Distribution**: Model prediction confidence
- **Sample Predictions**: Visual validation with confidence scores

---

## 💾 Model Usage

### Loading the Trained Model
```python
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the saved model
model = load_model('saved_models/animal_classifier_model.h5')

# Preprocess new image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Make prediction
image = preprocess_image('path/to/your/image.jpg')
predictions = model.predict(image)
class_idx = np.argmax(predictions)

# Class mapping
classes = ['cats', 'dogs', 'snakes']
predicted_class = classes[class_idx]
confidence = np.max(predictions)

print(f"Predicted: {predicted_class} (Confidence: {confidence:.3f})")
```

### Batch Prediction
```python
# For multiple images
images = [preprocess_image(path) for path in image_paths]
batch = np.vstack(images)
predictions = model.predict(batch)
```

---

## 🔧 Advanced Features

### Fine-tuning (Optional)
```python
# Unfreeze some top layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(optimizer=Adam(1e-5), 
             loss='categorical_crossentropy', 
             metrics=['accuracy'])
```

### Model Optimization
- **Quantization**: Reduce model size for mobile deployment
- **TensorFlow Lite**: Convert for mobile/edge inference
- **ONNX**: Export for cross-platform compatibility

---

## 🚀 Deployment Options

1. **Local Inference**: Use the saved .h5 model
2. **Web API**: Deploy with Flask/FastAPI
3. **Mobile Apps**: Convert to TensorFlow Lite
4. **Cloud**: Deploy on AWS/Google Cloud/Azure
5. **Edge Devices**: Optimize for Raspberry Pi/Jetson

---

## 🔍 Troubleshooting

### Common Issues
- **Low Accuracy**: Increase dataset size, add more augmentation
- **Overfitting**: Increase dropout, reduce model complexity
- **Memory Issues**: Reduce batch size, use gradient accumulation
- **Slow Training**: Use GPU, reduce image resolution

### Performance Tips
- Use GPU for training (`tensorflow-gpu`)
- Increase batch size if memory allows
- Use mixed precision training for speed
- Consider data pipeline optimization

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Applications](https://keras.io/api/applications/)

---

## 🙋‍♂️ Author

**Heshan Ranasinghe**  
Electronic and Telecommunication Engineering Undergraduate  
🌐 GitHub: [@DPHeshanRanasinghe](https://github.com/DPHeshanRanasinghe)  
📧 Email: [your-email@example.com]  
💼 LinkedIn: [Your LinkedIn Profile]

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## ⭐ Acknowledgments

- TensorFlow team for the excellent deep learning framework
- Google for MobileNetV2 architecture and ImageNet pretrained weights
- Open source community for inspiration and resources

---

## 🏷️ Tags

`#DeepLearning` `#ComputerVision` `#TensorFlow` `#Keras` `#TransferLearning` `#MobileNetV2` `#ImageClassification` `#CNN` `#MachineLearning` `#AI`