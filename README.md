🧠 Image Classifier using MobileNetV2 Transfer Learning

This project implements an image classification model using **MobileNetV2** and **transfer learning** with the **TensorFlow/Keras** framework. The dataset consists of labeled images of animals 
(such as cats, dogs, and snakes), and the goal is to accurately classify them using a pretrained convolutional neural network.

## 🚀 Project Features

- ✅ Transfer learning with **MobileNetV2 (ImageNet weights)**
- ✅ Data augmentation and preprocessing
- ✅ Training and validation using `ImageDataGenerator`
- ✅ Confusion matrix and classification report
- ✅ Model saving for future use
- ✅ Clean, modular structure and best practices

---

```
## 📂 Project Structure
Image-Classifier-MobileNetV2/
├── datasets
│   └── [Animals]  # Organized folders per class (e.g., cats, dogs, snakes)
│
├── notebooks
│   └── Project_1.ipynb   # Main training notebook
│
├── README.md         # Project overview
|   └── requirements.txt            # Python dependencies
|
├── saved_models
    └── animal_classifier_model.h5           # Saved model
```

---

## 🧰 Technologies Used

- Python 3.11.11
- TensorFlow / Keras
- scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 📊 Dataset

The dataset is organized in folders for each class. Example structure:
````
Animals
 ├── cats
 ├── dogs
 └── snakes
````

 ---

 ## 🏗️ How to Run

  1. Clone this repo:
  
  ```bash
  git clone https://github.com/DPHeshanRanasinghe/Image-Classifier-MobileNetV2.git
  cd Image-Classifier-MobileNetV2
  ```
  2. Set up your environment:
     pip install -r requirements.txt
  
  3. Run the notebook:
     Open in Google Colab or Jupyter Notebook
     Execute step-by-step to train and evaluate the model
     
      🧠 Results
      Final validation accuracy: ~98.8%
      Model generalized well with minimal overfitting
      High precision, recall, and F1-score across all classes
      Confusion Matrix:
      
      💾 Model Saving
      After training, the model is saved in .h5 format and can be loaded for inference without retraining:
      ```
      from tensorflow.keras.models import load_model
      model = load_model('saved_models/model.h5')
      ```

---

## 🙋‍♂️ Author
Heshan Ranasinghe\
Electronic and Telecommunication Engineering Undergraduate\
GitHub: [@DPHeshanRanasinghe](https://github.com/DPHeshanRanasinghe)


---

## 📜 License
This project is open source under the MIT License.







