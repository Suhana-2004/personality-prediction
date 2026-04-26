# Personality Prediction using Naive Bayes

## Overview
A machine learning model that predicts personality type (Introvert, 
Extrovert, or Ambivert) based on 29 behavioral traits, deployed 
as an interactive Streamlit web application.

## Model
- Algorithm: Gaussian Naive Bayes
- Classes: Introvert, Extrovert, Ambivert
- Features: 29 behavioral trait scores (scale 0-10)
- Accuracy: 99.68%

## Features
- Trait-based personality prediction
- Probability breakdown for each personality type
- Interactive sliders for all 29 traits
- Real-time prediction with Streamlit

## Dataset
- 350 samples per class (Introvert, Extrovert, Ambivert)
- 29 behavioral features including social energy, risk taking,
  leadership, empathy, creativity, and more

## Project Structure
```
├── app.py               # Streamlit web app (original)
├── app_simply.py        # Streamlit web app (enhanced UI)
├── check_classes.py     # Verify model classes
└── p.ipynb              # Model training notebook
```

pip install streamlit scikit-learn pandas numpy seaborn matplotlib
streamlit run app.py
