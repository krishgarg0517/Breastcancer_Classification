# ==========================================================
# 🎗️ Breast Cancer Detection and Analysis System
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# ----------------------------------------------------------
# 🌸 Page Configuration
# ----------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")
st.title("🎗️ Breast Cancer Detection & Segmentation System")
st.caption("AI-powered system for breast cancer prediction using text and image data.")

# ----------------------------------------------------------
# 🌈 Load Local CSS (Optional)
# ----------------------------------------------------------
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# ----------------------------------------------------------
# 🧠 Load U-Net CNN Model for Image Segmentation
# ----------------------------------------------------------
def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) /
                (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

try:
    unet_model = load_model("tumor_unet_optimized.h5",
                            custom_objects={"bce_dice_loss": bce_dice_loss, "dice_loss": dice_loss})
    st.sidebar.success("✅ U-Net Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"⚠️ Could not load U-Net model: {e}")
    unet_model = None

# ----------------------------------------------------------
# 🧮 Load Text & Image Models (Random Forest)
# ----------------------------------------------------------
try:
    rf_text = joblib.load("text_model.pkl")
    target_encoder = joblib.load("text_target_encoder.pkl")
    feature_encoders = joblib.load("text_feature_encoders.pkl")
    st.sidebar.success("✅ Text Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"⚠️ Could not load text model: {e}")
    rf_text, target_encoder, feature_encoders = None, None, None

try:
    rf_img = joblib.load("image_model.pkl")
    st.sidebar.success("✅ Image Model Loaded Successfully")
except Exception as e:
    st.sidebar.warning(f"⚠️ Could not load image classification model: {e}")
    rf_img = None

# ----------------------------------------------------------
# 🧭 Sidebar Navigation
# ----------------------------------------------------------
option = st.sidebar.radio("Select Model Type", ["Image Data Prediction", "Text Data Prediction"])

# ==========================================================
# 📊 TEXT DATA PREDICTION (Random Forest)
# ==========================================================
if option == "Text Data Prediction":
    st.header("📊 Predict Patient Status (Alive / Dead)")
    st.write("Enter the patient’s clinical details below:")

    if rf_text is not None:
        input_features = [
            "Age","Race","Marital Status" ,"T Stage ", "N Stage", "6th Stage",
            "differentiate", "Grade", "A Stage", "Tumor Size",
            "Estrogen Status", "Progesterone Status",
            "Regional Node Examined", "Reginol Node Positive", "Survival Months"
        ]

        user_input = {}
        for col in input_features:
            if col in feature_encoders.keys():
                options = list(feature_encoders[col].classes_)
                user_input[col] = st.selectbox(f"{col}", options)
            else:
                user_input[col] = st.number_input(f"{col}", min_value=0.0, step=1.0)

        if st.button("🔍 Predict Status"):
            try:
                input_df = pd.DataFrame([user_input])
                for col, enc in feature_encoders.items():
                    if col in input_df.columns:
                        input_df[col] = enc.transform(input_df[col])
                X_new = input_df.values
                pred_encoded = rf_text.predict(X_new)
                pred_label = target_encoder.inverse_transform(pred_encoded)[0]

                st.success(f"✅ Predicted Patient Status: **{pred_label}**")
                st.info("Model Used: Random Forest + SMOTE (Text Data)")
            except Exception as e:
                st.error(f"⚠️ Error during prediction: {e}")
    else:
        st.warning("⚠️ Text model files not found. Please check .pkl files.")

# ==========================================================
# 🩺 IMAGE DATA PREDICTION (CNN + Random Forest)
# ==========================================================
elif option == "Image Data Prediction":
    st.header("🩺 Tumor Detection & Classification from Ultrasound")

    uploaded_file = st.file_uploader("Upload an Ultrasound Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # --- Preprocess image for CNN ---
        img_eq = cv2.equalizeHist(img_gray)
        img_eq = cv2.bilateralFilter(img_eq, 7, 75, 75)
        img_resized = cv2.resize(img_eq, (256, 256)) / 255.0

        st.image(img_color[:, :, ::-1], caption="Uploaded Ultrasound", use_container_width=True)

        # --- Random Forest classification ---
        if rf_img is not None:
            img_rf = cv2.resize(img_color, (64, 64))
            pred_type = rf_img.predict(img_rf.flatten().reshape(1, -1))[0]
            label_map = {0: "Benign", 1: "Malignant", 2: "Normal"}
            st.success(f"🧠 Predicted Tumor Type: **{label_map.get(pred_type,'Unknown')}**")

        # --- U-Net segmentation ---
        if unet_model is not None:
            pred_mask = unet_model.predict(
                np.expand_dims(np.expand_dims(img_resized, 0), -1)
            )[0, :, :, 0]
            pred_mask = cv2.GaussianBlur(pred_mask, (3, 3), 0)
            mask_bin = (pred_mask > 0.4).astype(np.uint8) * 255

            kernel = np.ones((5, 5), np.uint8)
            mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result_img = cv2.cvtColor((img_resized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            if contours:
                best_cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(best_cnt)
                overlay = result_img.copy()
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
                result_img = cv2.addWeighted(overlay, 0.3, result_img, 0.7, 0)
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_img, "Tumor", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                st.image(result_img, caption="Detected Tumor Region", use_container_width=True)
            else:
                st.info("✅ No tumor region detected (clear image).")
        else:
            st.error("⚠️ CNN model not loaded. Check .h5 file path.")

# ==========================================================
# 🔚 Footer
# ==========================================================
st.markdown("---")
st.caption("Developed by Krish Garg 🎗️ | Deep Learning + Machine Learning Integrated Breast Cancer Detection System")


