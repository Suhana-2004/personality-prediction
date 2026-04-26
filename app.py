import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Configuration & Load Assets ---

# 🎨 Set a wide layout and a fun page title with an icon
st.set_page_config(
    page_title="Personality Predictor ✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 💡 Mapping with Emojis for better visual output
CLASS_MAPPING = {
    0: '🧠 Ambivert (Balanced)',
    1: ' extrovert (Outgoing)',
    2: ' Introvert (Reflective)'
}

# Function to load the pickled assets with error handling
@st.cache_resource
def load_assets():
    try:
        # Check if the required files exist
        if not os.path.exists('best_model.pkl'):
            st.error("Model file 'best_model.pkl' not found. Please ensure it is in the same directory.")
            return None, None
        if not os.path.exists('scalar.pkl'):
            st.error("Scaler file 'scalar.pkl' not found. Please ensure it is in the same directory.")
            return None, None

        # Load the saved model and scaler
        with open('best_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('scalar.pkl', 'rb') as scalar_file:
            scaler = pickle.load(scalar_file)

        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

# Load the model and scaler
model, scaler = load_assets()

FEATURE_COLUMNS = [
    'social_energy', 'alone_time_preference', 'talkativeness', 'deep_reflection', 
    'group_comfort', 'party_liking', 'listening_skill', 'empathy', 
    'creativity', 'organization', 'leadership', 'risk_taking', 
    'public_speaking_comfort', 'curiosity', 'routine_preference', 
    'excitement_seeking', 'friendliness', 'emotional_stability', 
    'planning', 'spontaneity', 'adventurousness', 'reading_habit', 
    'sports_interest', 'online_social_usage', 'travel_desire', 
    'gadget_usage', 'work_style_collaborative', 'decision_speed', 
    'stress_handling'
]

# --- Streamlit UI Components ---

# ⚙️ Sidebar for Context and Instructions
with st.sidebar:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR-qKz3-L7j-D-qQ9B3l3m-uQ8B6n-eA7oYQ&s") # Placeholder for an abstract image
    st.title("About the Predictor")
    st.markdown(
        """
        This application uses a trained **Naive Bayes Classifier** to predict an individual's personality type (Ambivert, Extrovert, or Introvert) 
        based on 29 key traits.
        
        **Instructions:**
        1.  Use the sliders on the main page to rate a person's traits on a scale of 0 (low) to 10 (high).
        2.  Click the **Predict Personality Type** button to get the result!
        """
    )
    st.markdown("---")
    st.info("The model and scaler were generated from your uploaded `p.ipynb` notebook.")

# 🖥️ Main Page Content
st.title("🔮 Personalized Trait-Based Personality Predictor")
st.markdown("---")

if model is None or scaler is None:
    st.error("Application setup failed. Please check the files `best_model.pkl` and `scalar.pkl`.")
    st.stop() 

st.subheader("1. Enter Trait Scores (Scale: 0 - 10)")
st.caption("Lower scores indicate a weaker presence of the trait, higher scores indicate a stronger presence.")

# Create an expander to hide the long list of sliders, improving initial view
with st.expander("📝 Click to input all 29 personality scores", expanded=True):
    col1, col2 = st.columns(2)
    user_inputs = {}

    def format_label(col_name):
        return col_name.replace('_', ' ').title()

    # Populate the user_inputs dictionary with slider values
    for i, col in enumerate(FEATURE_COLUMNS):
        if i % 2 == 0:
            with col1:
                user_inputs[col] = st.slider(
                    format_label(col),
                    min_value=0.0, 
                    max_value=10.0, 
                    value=5.0, 
                    step=0.1,
                    key=col
                )
        else:
            with col2:
                user_inputs[col] = st.slider(
                    format_label(col),
                    min_value=0.0, 
                    max_value=10.0, 
                    value=5.0, 
                    step=0.1,
                    key=col
                )

st.markdown("---")
st.subheader("2. Run Prediction")

# Prediction button
if st.button("✨ Predict Personality Type", type="primary"):
    # 1. Prepare the data
    input_df = pd.DataFrame([user_inputs], columns=FEATURE_COLUMNS)
    
    # 2. Scale the input data
    scaled_input = scaler.transform(input_df)
    
    # 3. Make the prediction (returns the numerical index: 0, 1, or 2)
    prediction_index = model.predict(scaled_input)[0]
    
    # 4. Map the index to the attractive class name with emoji
    prediction_class = CLASS_MAPPING.get(prediction_index, "❓ Error: Unknown Type")
    
    # 5. Display the result in a highly visible manner
    st.markdown("### **Prediction Result**")
    
    # Use st.success for a nice green box effect
    st.success(f"**Based on the inputs, the predicted personality type is:**")
    
    # Use st.markdown with a large, bold font for the final result
    st.markdown(f"## {prediction_class}")
    
    st.balloons()