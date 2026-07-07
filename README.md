# 🌸 CNN Flower Classification

A deep learning image classification project that uses a custom **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** to classify images into **five different flower species**: **Daisy, Dandelion, Rose, Sunflower, and Tulip**.

The project includes a modern **Streamlit** web application where users can upload a flower image and receive instant predictions with confidence scores.

---

## 🚀 Live Demo

👉 **(https://jswsguqfbghyinicifgrfm.streamlit.app/)**

---

## ✨ Features

- 🌸 Classifies 5 flower species
- 🧠 Custom CNN architecture
- 📷 Upload flower images for instant prediction
- 📊 Displays confidence scores for all classes
- ⚡ Fast inference with TensorFlow
- 🌐 Interactive Streamlit web application
- 💻 CPU-compatible deployment

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow
- Pandas
- Matplotlib

---

## 📂 Project Structure

```text
CNN-Flower-Classification/
│
├── app.py
├── models.h5
├── requirements.txt
├── README.md
└── sample_images/
```

---

## 📊 Dataset

This project was trained using the **Flowers Recognition** dataset from Kaggle.

**Dataset Link**

https://www.kaggle.com/datasets/alxmamaev/flowers-recognition

### Classes

- 🌼 Daisy
- 🌿 Dandelion
- 🌹 Rose
- 🌻 Sunflower
- 🌷 Tulip

---

## 🧠 Model Architecture

The model is a custom Convolutional Neural Network (CNN) designed for multi-class flower classification.

```text
Input Image (224×224)
        │
Image Preprocessing
        │
Convolution Layers
        │
Max Pooling
        │
Flatten
        │
Dense Layers
        │
Softmax
        │
5 Flower Classes
```

---

## 💾 Model Saving

The trained model was saved using TensorFlow's `.h5` format.

```python
model.save("models.h5")
```

The saved model is loaded during inference using:

```python
from tensorflow.keras.models import load_model

model = load_model("models.h5")
```

---

## 📈 Model Performance

| Metric | Value |
|---------|------:|
| Model | CNN |
| Framework | TensorFlow |
| Classes | 5 |
| Task | Flower Image Classification |
| Input Size | 224 × 224 |
| Output | Daisy, Dandelion, Rose, Sunflower, Tulip |

> Replace this section with your final validation accuracy if available.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/CNN-Flower-Classification.git

cd CNN-Flower-Classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🎯 Future Improvements

- Improve model accuracy with additional training
- Experiment with transfer learning (EfficientNet, MobileNetV2)
- Add Grad-CAM visualizations
- Deploy using Docker
- Support batch image predictions

---

## 📄 License

This project is intended for educational and research purposes.

---

## 👨‍💻 Author

**Mohamed Ahmed**

If you found this project useful, consider giving it a ⭐ on GitHub!
