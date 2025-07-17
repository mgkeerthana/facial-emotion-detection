# Facial Emotion Detection 🎭

This project detects facial emotions in real-time using deep learning models (CNN and EfficientNetB0), trained on the FER2013 dataset. The model is deployed in a Streamlit web app with webcam integration.

## 🔧 Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- OpenCV

## 🛠 Features
- Live webcam-based emotion prediction
- Transfer learning with EfficientNetB0
- Preprocessing: Grayscale, normalization, augmentation
- Metrics: Accuracy, confusion matrix, predictions

## 🚀 Run Locally

```bash
git clone https://github.com/keerthanamg/facial-emotion-detection
cd facial-emotion-detection
pip install -r requirements.txt
streamlit run app.py
