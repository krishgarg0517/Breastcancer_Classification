🎗️ Breast Cancer Detection & Segmentation System
Machine Learning + Deep Learning + Streamlit Deployment

This project is an end-to-end AI-based Breast Cancer Analysis System that combines Machine Learning, Deep Learning, and Web Deployment to assist in early diagnosis of breast cancer using both clinical data and ultrasound images.

🚀 Features

Clinical Data Prediction using Random Forest

Ultrasound Image Classification (baseline Random Forest model)

Tumor Segmentation using U-Net CNN

Bounding-Box Localization for clear tumor identification

Streamlit Web Application for real-time predictions

Modern UI with custom CSS

🧠 Models Used
🔹 Random Forest (Clinical Data)

Trained on Kaggle Breast Cancer dataset

Achieved 91.8% accuracy

🔹 Random Forest (Image Baseline)

Flattened 64×64 grayscale ultrasound images

Achieved 74.05% accuracy

🔹 U-Net CNN (Segmentation)

5-level encoder-decoder architecture

Dice Loss + BCE Loss

High-quality tumor segmentation

Automatic bounding-box generation

📊 Datasets
1. Clinical Dataset (Kaggle – Yasserh)

570 samples

30 numerical/categorical clinical features

Target: Benign / Malignant

2. BUSI Ultrasound Dataset (Kaggle – Aryashah2k)

780+ ultrasound images

Includes segmentation masks

Classes: Normal, Benign, Malignant
```
📂 Project Structure
├── app.py                 # Streamlit application
├── train_unet.py         # U-Net training script
├── main.ipynb            # Experiment notebook
├── Breast_Cancer.csv     # Clinical dataset
├── image_model.pkl       # RF image model
├── text_model.pkl        # RF text model
├── tumor_unet.h5         # U-Net segmentation model
├── style.css             # Custom UI styling
└── data/
    └── Dataset_BUSI_with_GT/
```

⚙️ Installation
```
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
streamlit run app.py
```
📈 Key Highlights

Hybrid ML + DL approach

Accurate segmentation & localization

Fully deployed, interactive Streamlit UI

Real-world, medical-use-case oriented project

Covers preprocessing → training → evaluation → deployment end-to-end
