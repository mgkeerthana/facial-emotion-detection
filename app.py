import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------- Config -------------------
IMG_SIZE = 224
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
MODEL_PATH = 'efficientnet_emotion_model.h5'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# ------------------- Load Model -------------------
@st.cache_resource
def load_emotion_model():
    return load_model(MODEL_PATH)

model = load_emotion_model()
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ------------------- Load Test Data for Evaluation -------------------
@st.cache_resource
def load_test_data():
    test_dir = "D:\\SP Project\\facial_emotion_app\\fer2013\\data\\test"
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='rgb',
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    return test_data

# ------------------- Evaluation Section -------------------
def evaluate_model():
    st.subheader("📊 Model Evaluation on Test Set")
    test_data = load_test_data()
    predictions = model.predict(test_data)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_data.classes
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=EMOTIONS, output_dict=True)

    st.markdown(f"✅ **Accuracy:** `{acc * 100:.2f}%`")
    st.markdown("📋 **Classification Report:**")
    st.dataframe(report)

    st.markdown("📌 **Confusion Matrix:**")
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt.gcf())

# ------------------- Prediction from Image -------------------
def predict_emotion_from_image(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No face detected.")
        return None

    for (x, y, w, h) in faces:
        roi = image_np[y:y+h, x:x+w]
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        roi = roi.astype("float32") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi)[0]
        emotion_idx = np.argmax(preds)
        label = EMOTIONS[emotion_idx]
        confidence = preds[emotion_idx]

        # Draw box and label
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image_np, f"{label} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    st.image(image_np, caption="Predicted Emotion", channels="RGB")

# ------------------- Webcam Utility -------------------
def use_webcam():
    st.warning("Webcam works only when run locally via: `streamlit run app.py`")
    st.info("Webcam support in Streamlit Cloud is limited. Use image upload here instead.")

# ------------------- Main Streamlit App -------------------
st.set_page_config(page_title="Facial Emotion Detector", layout="wide")
st.title("😃 Facial Emotion Detection using EfficientNetB0")

tab1, tab2, tab3 = st.tabs(["🔍 Detect Emotion", "📊 Evaluate Model", "🎥 Webcam"])

with tab1:
    st.header("🔍 Upload an Image to Detect Emotion")
    uploaded_file = st.file_uploader("Choose a face image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        st.image(image_np, caption="Original Image", channels="RGB")
        predict_emotion_from_image(image_np)

with tab2:
    evaluate_model()

with tab3:
    use_webcam()
