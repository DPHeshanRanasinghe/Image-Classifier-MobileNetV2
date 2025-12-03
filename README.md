# 🐾 Animal Image Classifier using MobileNetV2 Transfer Learning

This project implements a **high-performance deep learning image classification model** using **MobileNetV2** and **transfer learning** with the **TensorFlow/Keras** framework. The model accurately classifies animal images into three categories: **cats**, **dogs**, and **snakes**, achieving **98% validation accuracy** using a pretrained convolutional neural network with custom classification layers.

> **🎯 Achievement: 98% Validation Accuracy | 97.27% Average Prediction Confidence**

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
├── Animals/                              # Dataset folder
│   ├── cats/                            # 1,000 cat images
│   ├── dogs/                            # 1,000 dog images
│   └── snakes/                          # 1,000 snake images
│
├── Image-Classifier-MobileNetV2.ipynb   # Main training notebook (Jupyter)
│
├── animal_classifier_model.h5           # Trained Keras model (9.24 MB)
├── predictions.npy                      # Model predictions (900 samples)
├── true_classes.npy                     # Ground truth labels
├── predicted_classes.npy                # Predicted class indices
│
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
└── GitHub (Public) Repository.txt       # Repository link
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

## 📊 Dataset Information

The dataset is organized in the following folder structure:
```
Animals/
├── cats/       # 1,000 images of cats
├── dogs/       # 1,000 images of dogs
└── snakes/     # 1,000 images of snakes
```

**Dataset Statistics:**
- **Total Images**: 3,000 images
- **Classes**: 3 (cats, dogs, snakes)
- **Training Set**: 2,100 images (70%)
- **Validation Set**: 900 images (30%)
- **Image Size**: 224×224 pixels (automatically resized)
- **Format**: JPG, PNG supported
- **Distribution**: Balanced dataset (1,000 images per class)

---

## 🏗️ Installation & Usage

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/DPHeshanRanasinghe/Image-Classifier-MobileNetV2.git
cd Image-Classifier-MobileNetV2
```

### 2️⃣ **Set Up Python Environment**
```bash
# Create virtual environment (recommended)
python -m venv ml_env
# Activate environment
ml_env\Scripts\activate          # Windows
source ml_env/bin/activate       # Linux/Mac

# Install required packages
pip install tensorflow>=2.10.0
pip install numpy pandas matplotlib seaborn
pip install scikit-learn jupyter
```

### 3️⃣ **Prepare Dataset**
- Organize your images in the `Animals/` folder:
  ```
  Animals/
  ├── cats/      # Place cat images here
  ├── dogs/      # Place dog images here
  └── snakes/    # Place snake images here
  ```
- Recommended: At least 500+ images per class for good performance

### 4️⃣ **Launch Training**
```bash
# Start Jupyter Notebook
jupyter notebook Image-Classifier-MobileNetV2.ipynb

# Or use JupyterLab
jupyter lab
```

### 5️⃣ **Execute the Notebook**
Run cells sequentially:
1. **Section 1**: Import libraries
2. **Section 2**: Load and explore dataset
3. **Section 3**: Setup data preprocessing with augmentation
4. **Section 4**: Visualize sample images
5. **Section 5**: Create MobileNetV2 model
6. **Section 6**: Train the model (~7 minutes)
7. **Section 7**: Visualize training history
8. **Section 8**: Comprehensive evaluation
9. **Section 9**: Save trained model

### 6️⃣ **Use the Trained Model**
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load saved model
model = load_model('animal_classifier_model.h5')

# Prepare image
img = image.load_img('test_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)
class_names = ['cats', 'dogs', 'snakes']
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions) * 100

print(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")
```

---

## 🎯 Model Architecture

```
Input (224×224×3 RGB Image)
        ↓
┌─────────────────────────────────┐
│   MobileNetV2 Base (Frozen)     │
│   - ImageNet Pretrained Weights │
│   - 154 Layers                  │
│   - 2,257,984 Parameters        │
│   - Feature Extraction Only     │
└─────────────────────────────────┘
        ↓
GlobalAveragePooling2D
   (Spatial Dimensions → 1×1)
        ↓
Dense(128, ReLU)
   (Feature Learning Layer)
        ↓
Dropout(0.5)
   (Regularization)
        ↓
Dense(3, Softmax)
   (Classification Layer)
        ↓
Output [cats, dogs, snakes]
   (Probability Distribution)
```

**Model Summary:**
- **Total Parameters**: 2,422,339 (9.24 MB)
- **Trainable Parameters**: 164,355 (642.01 KB) - Only custom head
- **Non-trainable Parameters**: 2,257,984 (8.61 MB) - Frozen MobileNetV2 base

**Training Configuration:**
- **Optimizer**: Adam (learning rate = 0.0001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Max Epochs**: 15 (Early stopping enabled)
- **Actual Epochs**: 10 (Stopped early for best performance)
- **Callbacks**: 
  - EarlyStopping (patience=3, monitor='val_loss')
  - ReduceLROnPlateau (factor=0.2, patience=2)

**Data Augmentation (Training Only):**
- Image rescaling: 1/255
- Rotation range: ±20°
- Width/Height shift: ±20%
- Horizontal flip: Enabled
- Zoom range: ±20%
- Fill mode: Nearest

---

## 📈 Results & Performance

### 🎯 **Final Model Performance**
- ✅ **Validation Accuracy**: **98.00%**
- ✅ **Training Accuracy**: **97.95%**
- ✅ **Validation Loss**: 0.0692
- ✅ **Training Loss**: 0.0711
- ✅ **Training Status**: **Healthy (No Overfitting Detected)**
- ⚡ **Training Time**: ~7 minutes (10 epochs on GPU)
- 💾 **Model Size**: 9.24 MB (.h5 format)
- 🚀 **Inference Speed**: ~10ms per image (batch mode)

### 📊 **Per-Class Performance**
| Class  | Precision | Recall | F1-Score | Accuracy | Support |
|--------|-----------|--------|----------|----------|---------|
| **Cats**   | 1.0000    | 0.9500 | 0.9744   | 95.00%   | 300     |
| **Dogs**   | 0.9522    | 0.9967 | 0.9739   | 99.67%   | 300     |
| **Snakes** | 0.9900    | 0.9933 | 0.9917   | 99.33%   | 300     |

**Overall Metrics:**
- **Accuracy**: 98.00%
- **Macro Average**: 98.08%
- **Weighted Average**: 98.08%

### 🎯 **Confusion Matrix Analysis**
```
Actual → Predicted    Cats    Dogs    Snakes
─────────────────────────────────────────────
Cats                  285      13       2
Dogs                    0     299       1
Snakes                  0       2     298
```

**Key Insights:**
- ✅ Dogs classification is nearly perfect (99.67%)
- ✅ Snakes classification is excellent (99.33%)
- ⚠️ Cats have slight confusion with dogs (13 misclassifications)
- ✅ Zero cats/dogs misclassified as snakes
- ✅ Model shows strong discriminative features

### 📊 **Prediction Confidence Analysis**
- **Average Confidence**: 97.27%
- **Minimum Confidence**: 45.68%
- **Maximum Confidence**: 100.00%
- **High Confidence (>90%)**: ~780/900 predictions (86.7%)
- **Medium Confidence (70-90%)**: ~100/900 predictions (11.1%)
- **Low Confidence (<70%)**: ~20/900 predictions (2.2%)

### 📈 **Training Progress**
| Epoch | Train Acc | Val Acc | Train Loss | Val Loss | Learning Rate |
|-------|-----------|---------|------------|----------|---------------|
| 1     | 55.48%    | 95.89%  | 0.9901     | 0.2436   | 1.0e-04      |
| 2     | 91.37%    | 97.00%  | 0.2973     | 0.1311   | 1.0e-04      |
| 3     | 94.56%    | 97.22%  | 0.1755     | 0.0997   | 1.0e-04      |
| 5     | 95.17%    | 97.78%  | 0.1387     | 0.0817   | 1.0e-04      |
| 7     | 96.90%    | 98.00%  | 0.0925     | 0.0692   | 1.0e-04      |
| 10    | 98.00%    | 97.89%  | 0.0691     | 0.0724   | 2.0e-05      |

**Training Highlights:**
- 🚀 Rapid convergence in first 3 epochs
- 📉 Learning rate reduced at epoch 10 (ReduceLROnPlateau)
- ⚡ Early stopping triggered after epoch 10 (best weights restored)
- ✅ Consistent improvement with no overfitting

### 🎨 **Visualization Outputs**
1. **Training Curves**: Smooth convergence with minimal validation fluctuation
2. **Confusion Matrix**: Clear diagonal dominance showing accurate predictions
3. **Confidence Distribution**: Right-skewed distribution indicating high confidence
4. **Sample Predictions**: Visual validation with color-coded correctness (Green/Red)

---
## 🚀 Future Improvements

### Model Enhancements
- [ ] Fine-tune MobileNetV2 layers for improved accuracy
- [ ] Experiment with other architectures (EfficientNet, ResNet)
- [ ] Implement ensemble methods for better predictions
- [ ] Add more animal classes (birds, fish, etc.)
- [ ] Test with different image resolutions

### Deployment Options
- [ ] Create REST API using Flask/FastAPI
- [ ] Build web interface with Streamlit/Gradio
- [ ] Convert to TensorFlow Lite for mobile apps
- [ ] Deploy on AWS Lambda/Google Cloud Functions
- [ ] Create Docker container for easy deployment

### Advanced Features
- [ ] Implement Grad-CAM for visual explanations
- [ ] Add image preprocessing pipelines
- [ ] Create confidence threshold filtering
- [ ] Build real-time video classification
- [ ] Add multi-label classification support

---

## 🔍 Troubleshooting

### Common Issues & Solutions

#### ❌ **Low Validation Accuracy (<90%)**
- ✅ Increase dataset size (aim for 1000+ images per class)
- ✅ Add more data augmentation
- ✅ Train for more epochs
- ✅ Reduce learning rate
- ✅ Check for dataset quality/mislabeled images

#### ❌ **Overfitting (Train Acc >> Val Acc)**
- ✅ Increase dropout rate (try 0.6-0.7)
- ✅ Add more data augmentation
- ✅ Use L2 regularization
- ✅ Reduce model complexity
- ✅ Get more training data

#### ❌ **Memory Issues / OOM Errors**
- ✅ Reduce batch size (try 16 or 8)
- ✅ Use smaller image size (160×160)
- ✅ Enable mixed precision training
- ✅ Clear session between runs: `K.clear_session()`

#### ❌ **Slow Training Speed**
- ✅ Use GPU acceleration (CUDA + cuDNN)
- ✅ Increase batch size if memory allows
- ✅ Optimize data pipeline with `.cache()` and `.prefetch()`
- ✅ Use mixed precision training (float16)

#### ❌ **Model Not Loading**
- ✅ Check TensorFlow version compatibility
- ✅ Use `model.save('model.keras')` instead of `.h5`
- ✅ Save/load with custom objects: `load_model('model.h5', compile=False)`

### Performance Optimization Tips
- 🚀 Use GPU for 10-20x faster training
- 💾 Enable data caching for faster epoch iterations
- ⚡ Use mixed precision for 2-3x speedup
- 🎯 Monitor validation loss for early stopping
- 📊 Use TensorBoard for real-time monitoring

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
*Electronic and Telecommunication Engineering Undergraduate*

📧 **Email**: hranasinghe505@gmail.com  
🌐 **GitHub**: [@DPHeshanRanasinghe](https://github.com/DPHeshanRanasinghe)  
💼 **LinkedIn**: [Heshan Ranasinghe](https://www.linkedin.com/in/heshan-ranasinghe-988b00290)  
📍 **Location**: Sri Lanka

### 🎓 About
Passionate about **Deep Learning**, **Computer Vision**, and **AI Engineering**. This project demonstrates practical implementation of transfer learning for real-world image classification tasks with production-ready code quality.

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**You are free to:**
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ Sublicense

**Conditions:**
- 📝 Include original license and copyright notice
- ⚠️ No warranty provided 

---

## ⭐ Acknowledgments

- **TensorFlow Team** for the excellent deep learning framework
- **Google Research** for MobileNetV2 architecture and ImageNet pretrained weights
- **Keras Team** for intuitive high-level API
- **Open Source Community** for inspiration and resources
- **Contributors** who helped improve this project

---

## 📊 Project Statistics

- ⭐ **Stars**: Give this project a star if you find it helpful!
- 🐛 **Issues**: 0 open issues
- 📝 **Commits**: Regular updates and improvements
- 📦 **Dependencies**: Minimal and well-maintained
- 📄 **Documentation**: Comprehensive README with examples

---

## 🏷️ Keywords & Tags

`Deep Learning` • `Computer Vision` • `TensorFlow` • `Keras` • `Transfer Learning` • `MobileNetV2` • `Image Classification` • `CNN` • `Machine Learning` • `AI` • `Neural Networks` • `Python` • `Jupyter Notebook` • `Data Science` • `Animal Recognition` • `ImageNet` • `Model Training` • `Prediction`

---

## 📞 Support & Contact

💬 **Questions or Issues?**
- Open an [Issue](https://github.com/DPHeshanRanasinghe/Image-Classifier-MobileNetV2/issues)
- Email: hranasinghe505@gmail.com
- LinkedIn: [Message me](https://www.linkedin.com/in/heshan-ranasinghe-988b00290)

⭐ **Found this helpful?** Give it a star on GitHub!

🤝 **Want to contribute?** Pull requests are welcome!

---

<div align="center">

**Made with ❤️ by Heshan Ranasinghe**

*"Building AI solutions one model at a time"*

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/DPHeshanRanasinghe)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/heshan-ranasinghe-988b00290)

</div>