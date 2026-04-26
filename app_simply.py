import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Configuration & Page Setup ---
st.set_page_config(
    page_title="Personality AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    /* Main Title Styling */
    .main-title {
        text-align: center;
        font-size: 3em;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF512F, #DD2476);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        margin-bottom: 30px;
    }
    /* Result Card Styling */
    .result-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .result-text {
        font-size: 2.5em;
        font-weight: 900;
        margin: 0;
    }
    .result-label {
        font-size: 1.2em;
        font-weight: 500;
        opacity: 0.9;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

# --- Constants & Mappings ---
TRAIT_GROUPS = {
    "🤝 Social Energy & Interaction": [
        'social_energy', 'alone_time_preference', 'group_comfort', 
        'party_liking', 'talkativeness', 'public_speaking_comfort', 
    ],
    "🚀 Drive, Risk & Cognition": [
        'excitement_seeking', 'risk_taking', 'leadership', 'reading_habit'
    ]
}

FEATURE_COLUMNS = [
    'social_energy', 'alone_time_preference', 'talkativeness', 'group_comfort', 
    'party_liking', 'leadership', 'risk_taking', 'public_speaking_comfort', 
    'excitement_seeking', 'reading_habit'
]

CLASS_MAPPING = {
    0: {'name': 'Ambivert', 'label': '🧠 Ambivert (Balanced)', 'color': '#28a745', 'bg': '#d4edda'},
    1: {'name': 'Extrovert', 'label': '✨ Extrovert (Outgoing)', 'color': '#fd7e14', 'bg': '#ffe8cc'},
    2: {'name': 'Introvert', 'label': '🧘 Introvert (Reflective)', 'color': '#6f42c1', 'bg': '#e2d9f3'}
}
CLASS_NAMES = ['Ambivert', 'Extrovert', 'Introvert'] # Sorted names for probability lookup

EXPLANATIONS = {
    0: "Ambiverts possess a balanced mix of extroverted and introverted tendencies. They can enjoy social interaction and quiet reflection equally, adapting their behavior based on the situation.",
    1: "Extroverts gain energy from social interaction and external activities. They are typically outgoing, talkative, and comfortable in large groups, often enjoying excitement and risk-taking.",
    2: "Introverts gain energy from solitude and internal reflection. They tend to prefer smaller groups or one-on-one interactions, enjoying deep thought and less stimulating environments."
}

# --- Load Assets ---
@st.cache_resource
def load_assets():
    try:
        model_path = 'best_model.pkl' 
        scaler_path = 'scalar.pkl'     

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, None

        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(scaler_path, 'rb') as scalar_file:
            scaler = pickle.load(scalar_file)
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

model, scaler = load_assets()

# --- Sidebar Info ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4825/4825038.png", width=100)
    st.title("About Model")
    st.info(
        """
        This AI model analyzes **10 key behavioral traits** to classify personality into three categories.
        
        **Adjust the sliders** to see how changes in social energy or risk-taking affect the prediction.
        """
    )
    st.markdown("---")
    st.caption("Powered by Scikit-Learn & Streamlit")

# --- Main UI ---

if model is None or scaler is None:
    st.error("⚠️ **Missing Files:** Please ensure `best_model.pkl` and `scalar.pkl` are in the same folder as this script.")
    st.stop() 

# Title Section
st.markdown('<div class="main-title">Explainable Personality AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Adjust the sliders below (0 = Low, 10 = High) to analyze the personality profile.</div>', unsafe_allow_html=True)

user_inputs = {}

def format_label(col_name):
    return col_name.replace('_', ' ').title()

# --- Input Section (Two Columns Layout) ---
col_left, col_right = st.columns(2, gap="medium")

# Helper to distribute groups across columns
groups = list(TRAIT_GROUPS.items())

# Left Column Inputs
with col_left:
    group_name, traits = groups[0]
    st.subheader(group_name)
    valid_traits = [t for t in traits if t in FEATURE_COLUMNS]
    for col in valid_traits:
        user_inputs[col] = st.slider(
            format_label(col), 0.00, 10.00, 5.00, 0.01, key=col
        )

# Right Column Inputs
with col_right:
    group_name, traits = groups[1]
    st.subheader(group_name)
    valid_traits = [t for t in traits if t in FEATURE_COLUMNS]
    for col in valid_traits:
        user_inputs[col] = st.slider(
            format_label(col), 0.00, 10.00, 5.00, 0.01, key=col
        )

st.markdown("---")

# --- Prediction Section ---
# Centering the button
b1, b2, b3 = st.columns([1, 2, 1])
with b2:
    analyze_btn = st.button("🚀 Analyze Personality Profile", type="primary", use_container_width=True)

if analyze_btn:
    # 1. Prepare Data
    ALL_COLUMNS = scaler.feature_names_in_.tolist()
    input_29_df = pd.DataFrame([[5.0] * len(ALL_COLUMNS)], columns=ALL_COLUMNS)
    
    for col, value in user_inputs.items():
        input_29_df.loc[0, col] = value

    # 2. Scale & Filter
    scaled_input_29 = scaler.transform(input_29_df)
    feature_indices = [ALL_COLUMNS.index(col) for col in FEATURE_COLUMNS]
    scaled_input_10 = scaled_input_29[:, feature_indices]
    
    # 3. Predict
    prediction_index = model.predict(scaled_input_10)[0]
    prediction_proba_array = model.predict_proba(scaled_input_10)[0]
    
    # 4. Get Details
    result_info = CLASS_MAPPING.get(prediction_index, {'name': 'Unknown', 'label': 'Unknown', 'color': '#333', 'bg': '#eee'})
    explanation = EXPLANATIONS.get(prediction_index, "No explanation available.")
    
    # --- DISPLAY RESULTS ---
    
    # A. The Big Bold Banner
    st.markdown(f"""
        <div class="result-box" style="background-color: {result_info['bg']}; border: 2px solid {result_info['color']};">
            <p class="result-label" style="color: {result_info['color']};">PREDICTED TYPE</p>
            <p class="result-text" style="color: {result_info['color']};">{result_info['label']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # B. Explanation
    st.info(f"**💡 Insight:** {explanation}")
    if prediction_index == 2:
        st.caption("Note: Being an Introvert is a preference for how you recharge, not a measure of social competence.")

    st.markdown("### 📊 Probability Breakdown")
    
    # C. Probability Metrics
    proba_data = {name: prediction_proba_array[i] * 100 for i, name in enumerate(CLASS_NAMES)}
    sorted_probas = sorted(proba_data.items(), key=lambda item: item[1], reverse=True)

    # Create 3 columns for the metrics
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric(label=sorted_probas[0][0], value=f"{sorted_probas[0][1]:.1f}%", delta="Primary Match")
        st.progress(sorted_probas[0][1] / 100)
        
    with m2:
        st.metric(label=sorted_probas[1][0], value=f"{sorted_probas[1][1]:.1f}%")
        st.progress(sorted_probas[1][1] / 100)
        
    with m3:
        st.metric(label=sorted_probas[2][0], value=f"{sorted_probas[2][1]:.1f}%")
        st.progress(sorted_probas[2][1] / 100)

    st.balloons()