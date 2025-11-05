⚠️ Note: This branch (streamlit_app) contains the Streamlit frontend and exploratory data analysis (EDA) parts of the project.
The backend (FastAPI, Docker, and deployment files) are located in the main branch.

### 🧠 Social Anxiety Prediction
## 📋 Overview

This project aims to predict the level of social anxiety using a machine learning model trained on an open-source dataset from Kaggle.
The goal is to raise awareness about mental health and demonstrate how data-driven methods can help in understanding social anxiety patterns.

## 🎯 Objectives

1. Analyze and preprocess real-world mental health data

2. Build and evaluate machine learning models to predict social anxiety levels

3. Deploy an interactive web app for real-time predictions

## 🧩 Dataset

- Source: Kaggle (Open-source dataset)

- Size: ~11,000 rows and 18 features

- Features: Include demographic info, behavioral traits, and anxiety indicators

🧹 Data Preprocessing

- Outlier detection and removal

- Handling categorical data with encoding

- Train-test split for model validation

## 🤖 Model Training

Tested multiple algorithms using cross-validation and hyperparameter tuning

Best Model: CatBoostRegressor

Performance Metric: RMSE = 0.99

## 🌟 Key Insights  
The model predicts a user's **social anxiety level** as **High**, **Medium**, or **Low** based on their responses.  
This helps demonstrate how behavioral and demographic factors can influence social anxiety patterns.


## ☁️ Deployment
- Backend

- Built with FastAPI

- Dockerized and pushed to Google Artifact Registry

- Deployed on Google Cloud Run

- Frontend

- Built with Streamlit

- Connected to the FastAPI backend for real-time prediction

- Deployed to the cloud

### 🔗 **Live Demo:** [Social Anxiety App](https://social-anxiety-rtvr7zx2immck3kq8ycnpw.streamlit.app/)


## 👩‍💻 My Contribution

This project was completed as a team collaboration.
My contributions included:

- Performing Exploratory Data Analysis (EDA) to understand data distribution and detect outliers

- Handling data preprocessing and feature encoding

- Training and evaluating the machine learning models

- Selecting and fine-tuning the final CatBoost model

## 🧰 Tech Stack

- Languages: Python

- Libraries: pandas, NumPy, scikit-learn, CatBoost

- Frameworks: FastAPI, Streamlit

- Cloud: Google Cloud Run, Docker

## 📎 Files
```
streamlit_app/
│
├── api/                            # Contains API connection and related scripts
├── logic/                          # Core logic for the prediction and preprocessing
├── social_anxiety/                 # Streamlit frontend components
├── enhanced_anxiety_dataset.csv    # Cleaned dataset used for model and EDA
├── SOCIAL_ANXIETY_LEVEL_EDA.ipynb  # Jupyter notebook for data analysis and visualization
├── Home.py                         # Main Streamlit app file
├── .gitignore                      # Git ignore rules
├── .python-version                 # Python version tracking
├── .DS_Store                       # System file (can be ignored)
└── README.md                       # Project documentation
```

🗂️ Note: Backend and deployment files (FastAPI, Docker, Google Cloud) are located in the main branch.

## 📫 Contact  
This project was developed as part of a **team collaboration** focused on mental health awareness through data science.
