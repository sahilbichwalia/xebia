import streamlit as st
import pandas as pd
import joblib
import numpy as np
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from src.config.logger import setup_logging 

# Initialize logger
logger = setup_logging()

# Configure page
st.set_page_config(
    page_title="Pharmaceutical Safety Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger.info("Streamlit application started")

# Load models
@st.cache_resource
def load_models():
    """Load machine learning models with logging"""
    logger.info("Attempting to load models...")
    try:
        model = joblib.load('best_random_forest_model.pkl')
        encoder = joblib.load('label_encoder.pkl')
        logger.info("Models loaded successfully")
        logger.info(f"Model classes: {getattr(model, 'classes_', 'N/A')}")
        logger.info(f"Encoder classes count: {len(getattr(encoder, 'classes_', []))}")
        return model, encoder
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        st.error(f"Model files not found: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error(f"Error loading models: {e}")
        return None, None

# Main app
def main():
    """Main application function with logging"""
    logger.info("Main application function called")
    
    st.title("💊 Pharmaceutical Safety Predictor")
    st.markdown("### Predict drug safety based on pharmaceutical parameters")
    
    # Load models
    model, encoder = load_models()
    
    if model is None or encoder is None:
        logger.error("Models not loaded properly, stopping execution")
        st.error("Please ensure 'best_random_forest_model.pkl' and 'label_encoder.pkl' are in the same directory.")
        return
    
    # Sidebar for input parameters
    st.sidebar.header("Input Parameters")
    
    # Active Ingredient selection
    st.sidebar.subheader("Active Ingredient")
    
    # Get available active ingredients from encoder
    try:
        available_ingredients = encoder.classes_
        logger.info(f"Available ingredients loaded: {len(available_ingredients)} items")
        
        active_ingredient = st.sidebar.selectbox(
            "Select Active Ingredient:",
            options=available_ingredients,
            help="Choose the active pharmaceutical ingredient"
        )
        logger.info(f"User selected active ingredient: {active_ingredient}")
        
    except Exception as e:
        logger.error(f"Error loading active ingredients from encoder: {e}")
        st.sidebar.error("Error loading active ingredients from encoder")
        active_ingredient = st.sidebar.text_input("Enter Active Ingredient:")
        logger.info(f"User entered custom active ingredient: {active_ingredient}")
    
    # Numerical parameters
    st.sidebar.subheader("Physical & Chemical Properties")
    
    days_until_expiry = st.sidebar.number_input(
        "Days Until Expiry:",
        min_value=0,
        max_value=3650,
        value=365,
        help="Number of days until the drug expires"
    )
    
    storage_temp = st.sidebar.number_input(
        "Storage Temperature (°C):",
        min_value=-20.0,
        max_value=50.0,
        value=25.0,
        step=0.1,
        help="Recommended storage temperature"
    )
    
    warning_labels = st.sidebar.number_input(
        "Warning Labels Present:",
        min_value=0,
        max_value=10,
        value=1,
        help="Number of warning labels on the package"
    )
    
    dissolution_rate = st.sidebar.slider(
        "Dissolution Rate (%):",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=0.1,
        help="Rate at which the drug dissolves"
    )
    
    disintegration_time = st.sidebar.number_input(
        "Disintegration Time (minutes):",
        min_value=0.0,
        max_value=120.0,
        value=15.0,
        step=0.1,
        help="Time for tablet to disintegrate"
    )
    
    impurity_level = st.sidebar.slider(
        "Impurity Level (%):",
        min_value=0.0,
        max_value=10.0,
        value=0.5,
        step=0.01,
        help="Percentage of impurities in the drug"
    )
    
    assay_purity = st.sidebar.slider(
        "Assay Purity (%):",
        min_value=80.0,
        max_value=100.0,
        value=99.0,
        step=0.1,
        help="Purity of the active ingredient"
    )
    
    # Log input parameters
    input_params = {
        'active_ingredient': active_ingredient,
        'days_until_expiry': days_until_expiry,
        'storage_temp': storage_temp,
        'warning_labels': warning_labels,
        'dissolution_rate': dissolution_rate,
        'disintegration_time': disintegration_time,
        'impurity_level': impurity_level,
        'assay_purity': assay_purity
    }
    logger.info(f"Input parameters: {input_params}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Summary")
        
        # Create input dataframe
        input_data = {
            'Active Ingredient': [active_ingredient],
            'Days Until Expiry': [days_until_expiry],
            'Storage Temperature (°C)': [storage_temp],
            'Warning Labels Present': [warning_labels],
            'Dissolution Rate (%)': [dissolution_rate],
            'Disintegration Time (minutes)': [disintegration_time],
            'Impurity Level (%)': [impurity_level],
            'Assay Purity (%)': [assay_purity]
        }
        
        input_df = pd.DataFrame(input_data)
        st.dataframe(input_df, use_container_width=True)
        
        # Prediction button
        if st.button("🔮 Predict Safety", type="primary", use_container_width=True):
            logger.info("Prediction button clicked")
            logger.info(f"Making prediction for: {input_params}")
            
            try:
                # Prepare data for prediction
                X_input = input_df.copy()
                
                # Encode active ingredient
                try:
                    encoded_ingredient = encoder.transform([active_ingredient])
                    X_input['Active Ingredient'] = encoded_ingredient
                    logger.info(f"Active ingredient encoded successfully: {active_ingredient} -> {encoded_ingredient[0]}")
                    
                except ValueError as e:
                    logger.error(f"Active ingredient encoding failed: {active_ingredient} - {e}")
                    st.error(f"Active ingredient '{active_ingredient}' not found in training data")
                    return
                
                # Make prediction
                logger.info("Making model prediction...")
                prediction = model.predict(X_input)
                prediction_proba = model.predict_proba(X_input)
                
                logger.info(f"Prediction result: {prediction[0]}")
                logger.info(f"Prediction probabilities: {prediction_proba[0]}")
                
                # Display results
                st.subheader("Prediction Results")
                
                # Safety status
                safety_status = prediction[0]
                confidence = np.max(prediction_proba[0]) * 100
                
                logger.info(f"Final prediction: {safety_status} with confidence: {confidence:.1f}%")
                
                if safety_status == 'Safe':
                    st.success(f"✅ **SAFE** (Confidence: {confidence:.1f}%)")
                    logger.info(f"Drug classified as SAFE with {confidence:.1f}% confidence")
                else:
                    st.error(f"❌ **NOT SAFE** (Confidence: {confidence:.1f}%)")
                    logger.warning(f"Drug classified as NOT SAFE with {confidence:.1f}% confidence")
                
                # Probability breakdown
                st.subheader("Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Safety Status': model.classes_,
                    'Probability': prediction_proba[0] * 100
                })
                
                logger.info(f"Probability breakdown: {prob_df.to_dict()}")
                st.bar_chart(prob_df.set_index('Safety Status'))
                
            except Exception as e:
                logger.error(f"Error making prediction: {str(e)}", exc_info=True)
                st.error(f"Error making prediction: {str(e)}")
    
    with col2:
        st.subheader("Safety Guidelines")
        
        # Safety indicators
        st.markdown("#### 🟢 Good Indicators")
        st.markdown("""
        - Assay Purity > 95%
        - Impurity Level < 2%
        - Dissolution Rate > 75%
        - Proper storage temperature
        - Adequate shelf life
        """)
        
        st.markdown("#### 🔴 Risk Factors")
        st.markdown("""
        - Low purity levels
        - High impurity content
        - Poor dissolution
        - Temperature abuse
        - Near expiry
        """)
        
        # Model info
        st.subheader("Model Information")
        st.info("""
        **Model**: Random Forest Classifier
        
        **Features**: 8 pharmaceutical parameters
        
        **Purpose**: Predict drug safety based on quality control metrics
        """)

# Batch prediction feature
def batch_prediction():
    """Batch prediction function with logging"""
    logger.info("Batch prediction function called")
    
    st.header("📊 Batch Prediction")
    st.markdown("Upload a CSV file with multiple samples for batch prediction")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV should contain columns: Active Ingredient, Days Until Expiry, Storage Temperature (°C), Warning Labels Present, Dissolution Rate (%), Disintegration Time (minutes), Impurity Level (%), Assay Purity (%)"
    )
    
    if uploaded_file is not None:
        logger.info(f"File uploaded: {uploaded_file.name}")
        
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            logger.info(f"CSV loaded successfully. Shape: {df.shape}")
            logger.info(f"CSV columns: {df.columns.tolist()}")
            
            st.subheader("Uploaded Data")
            st.dataframe(df)
            
            # Load models
            model, encoder = load_models()
            
            if model is not None and encoder is not None:
                if st.button("Predict All", type="primary"):
                    logger.info("Batch prediction started")
                    logger.info(f"Processing {len(df)} samples")
                    
                    # Make predictions
                    df_pred = df.copy()
                    
                    # Encode active ingredients
                    try:
                        original_ingredients = df_pred['Active Ingredient'].tolist()
                        df_pred['Active Ingredient'] = encoder.transform(df_pred['Active Ingredient'])
                        logger.info(f"Active ingredients encoded successfully for batch prediction")
                        
                    except ValueError as e:
                        logger.error(f"Error encoding active ingredients in batch: {e}")
                        st.error(f"Error encoding active ingredients: {e}")
                        return
                    
                    # Predict
                    logger.info("Making batch predictions...")
                    predictions = model.predict(df_pred)
                    probabilities = model.predict_proba(df_pred)
                    
                    # Log prediction results
                    safe_count = sum(predictions == 'Safe')
                    not_safe_count = len(predictions) - safe_count
                    logger.info(f"Batch prediction results: {safe_count} Safe, {not_safe_count} Not Safe")
                    
                    # Add results to dataframe
                    df['Predicted Safety'] = predictions
                    df['Confidence (%)'] = np.max(probabilities, axis=1) * 100
                    
                    logger.info("Batch prediction completed successfully")
                    
                    st.subheader("Prediction Results")
                    st.dataframe(df)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name=f"safety_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    logger.info("Results prepared for download")
                    
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            st.error(f"Error processing file: {str(e)}")
