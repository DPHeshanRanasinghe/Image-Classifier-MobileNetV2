# Animal Image Classifier with MobileNetV2 Transfer Learning

A deep learning image classification system implementing MobileNetV2 transfer learning to classify animal images across three categories: cats, dogs, and snakes, achieving 98% validation accuracy.

## Performance Metrics

- **Validation Accuracy**: 98.00%
- **Training Accuracy**: 97.95%
- **Average Prediction Confidence**: 97.27%
- **Model Size**: 9.24 MB
- **Training Time**: ~7 minutes on GPU

## Key Features

- Transfer learning with ImageNet-pretrained MobileNetV2
- Comprehensive data augmentation pipeline
- Early stopping and learning rate scheduling
- Complete evaluation suite with confusion matrix
- Production-ready code structure

## Technical Stack

- **Framework**: TensorFlow 2.x / Keras
- **Architecture**: MobileNetV2
- **Language**: Python 3.8+
- **Libraries**: scikit-learn, NumPy, Matplotlib, Seaborn

## Project Structure

```
Image-Classifier-MobileNetV2/
├── Animals/                              
│   ├── cats/                            
│   ├── dogs/                            
│   └── snakes/                          
├── Image-Classifier-MobileNetV2.ipynb   
├── animal_classifier_model.h5           
├── predictions.npy                      
├── true_classes.npy                     
├── predicted_classes.npy                
├── README.md                            
└── requirements.txt                     
```

## Dataset

- **Total Images**: 3,000 (1,000 per class)
- **Training Set**: 2,100 images (70%)
- **Validation Set**: 900 images (30%)
- **Image Resolution**: 224×224 pixels
- **Format**: JPG, PNG

## Installation

Clone the repository:
```bash
git clone https://github.com/DPHeshanRanasinghe/Image-Classifier-MobileNetV2.git
cd Image-Classifier-MobileNetV2
```

Create virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Required packages:
```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

## Usage

### Training

1. Organize dataset:
```
Animals/
├── cats/
├── dogs/
└── snakes/
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook Image-Classifier-MobileNetV2.ipynb
```

3. Execute cells sequentially to train and evaluate the model

### Making Predictions

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model('animal_classifier_model.h5')

# Prepare image
img = image.load_img('path/to/image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
class_labels = ['cats', 'dogs', 'snakes']
predicted_class = class_labels[np.argmax(predictions)]
confidence = np.max(predictions) * 100

print(f"Predicted: {predicted_class} ({confidence:.2f}%)")
```

## Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 Base (Frozen) - 2,257,984 params
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(3, Softmax)
```

**Configuration:**
- Total Parameters: 2,422,339
- Trainable: 164,355 (classification head only)
- Optimizer: Adam (lr=0.0001)
- Batch Size: 32
- Epochs: 10 (early stopping)

## Results

### Classification Performance

| Class  | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Cats   | 1.0000    | 0.9500 | 0.9744   |
| Dogs   | 0.9522    | 0.9967 | 0.9739   |
| Snakes | 0.9900    | 0.9933 | 0.9917   |

**Overall Accuracy**: 98.00%

### Confusion Matrix

```
Actual → Predicted    Cats    Dogs    Snakes
Cats                  285      13       2
Dogs                    0     299       1
Snakes                  0       2     298
```

## Troubleshooting

**Low Accuracy:**
- Ensure at least 500 images per class
- Verify correct labeling
- Increase training epochs

**Memory Issues:**
- Reduce batch size to 16 or 8
- Use smaller input size (160×160)

**Slow Training:**
- Use GPU acceleration
- Enable mixed precision training

## Future Enhancements

- Fine-tune deeper layers
- Expand to more animal categories
- REST API deployment
- Mobile app with TensorFlow Lite

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Submit Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

## Author

**Heshan Ranasinghe**  
Electronic and Telecommunication Engineering Undergraduate

- Email: hranasinghe505@gmail.com
- GitHub: [@DPHeshanRanasinghe](https://github.com/DPHeshanRanasinghe)
- LinkedIn: [Heshan Ranasinghe](https://www.linkedin.com/in/heshan-ranasinghe-988b00290)

## Citation

```bibtex
@software{ranasinghe2024animalclassifier,
  author = {Ranasinghe, Heshan},
  title = {Animal Image Classifier with MobileNetV2 Transfer Learning},
  year = {2024},
  url = {https://github.com/DPHeshanRanasinghe/Image-Classifier-MobileNetV2}
}
```