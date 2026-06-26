import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from ast import literal_eval
import matplotlib.pyplot as plt
import uuid
from datetime import datetime

from data_processor import DataProcessor
from utils import get_model_path, load_model, symptoms_to_vector
from single_symptom_mapping import load_symptom_mapping, generate_single_symptom_mapping
from database import Database

from model_trainer import train_model
from data_preprocessing import preprocess_pipeline, process_user_input, get_all_symptoms

st.set_page_config(
    page_title="Dendrites: Personalized Medication Recommender",
    page_icon="https://cdn-icons-png.flaticon.com/512/2867/2867378.png",
    layout="wide"
)

if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = []

if 'show_results' not in st.session_state:
    st.session_state.show_results = False

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None

if 'recommendation_engine' not in st.session_state:
    st.session_state.recommendation_engine = None

if 'single_symptom_mapping' not in st.session_state:
    st.session_state.single_symptom_mapping = None

if 'model' not in st.session_state:
    st.session_state.model = None

if 'feature_metadata' not in st.session_state:
    st.session_state.feature_metadata = None

if 'clean_data' not in st.session_state:
    st.session_state.clean_data = None

if 'db' not in st.session_state:
    st.session_state.db = Database()

if 'recommendation_id' not in st.session_state:
    st.session_state.recommendation_id = None

if 'show_history' not in st.session_state:
    st.session_state.show_history = False

st.title("Dendrites: Personalized Medication Recommender")
st.markdown("### Get medication recommendations based on your symptoms")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2867/2867378.png", width=100)
    st.header("About Dendrites")
    st.info(
        "This application uses machine learning (specifically random forest classifier) to recommend medications based on your symptoms. "
        "Simply select your symptoms and receive personalized medication recommendations."
    )

    st.link_button("View Source Code", "https://github.com/thestaccato/dendrites")

    if st.button("View History", key="view_history_btn"):
        st.session_state.show_history = True
        st.session_state.show_results = False

    if st.button("New Recommendation", key="new_rec_btn"):
        st.session_state.show_history = False
        st.session_state.show_results = False

    st.markdown("### How it works")
    st.markdown(
        "1. Select your symptoms from the dropdown menu\n"
        "2. Click 'Get Recommendations'\n"
        "3. View your personalized medication recommendations\n"
        "4. Check the confidence score to understand the reliability"
    )

    st.warning(
        "**Disclaimer**: This app is a prototype built for educational purposes and should not replace "
        "professional medical advice. Always consult with a healthcare provider before "
        "taking any medication."
    )

    st.markdown("---")

    if st.checkbox("Show System Statistics", key="show_stats"):
        try:
            popular_symptoms = st.session_state.db.get_popular_symptoms(10)
            if popular_symptoms:
                st.subheader("Popular Symptoms")
                for symptom, count in popular_symptoms:
                    st.write(f"- {symptom}: {count} occurrences")
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")

    st.markdown("© 2026 Yash Sharma | [BSD-3-Clause](https://github.com/thestaccato/dendrites/blob/main/LICENSE)")

@st.cache_resource
def initialize_system():
    """Initialize the data processor, model, and single symptom mapping"""
    data_processor = DataProcessor('attached_assets/merged_health_dataset.csv')

    mapping_path = 'models/single_symptom_mapping.pkl'
    if not os.path.exists(mapping_path):
        single_symptom_mapping = generate_single_symptom_mapping(df=data_processor.df, output_path=mapping_path)
    else:
        single_symptom_mapping = load_symptom_mapping(mapping_path)

    model_path = get_model_path()
    model = None
    label_encoder = None
    all_symptoms = None
    symptom_to_index = None

    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
                if isinstance(model_data, dict) and 'model' in model_data and 'symptom_to_index' in model_data:
                    model = model_data['model']
                    label_encoder = model_data['label_encoder']
                    all_symptoms = model_data['all_symptoms']
                    symptom_to_index = model_data['symptom_to_index']
                else:
                    print("Old model format detected, retraining...")
                    model, label_encoder, all_symptoms = train_model()
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        model = model_data['model']
                        label_encoder = model_data['label_encoder']
                        all_symptoms = model_data['all_symptoms']
                        symptom_to_index = model_data['symptom_to_index']
                        
        except Exception as e:
            print(f"Error loading model: {e}")
            model, label_encoder, all_symptoms = train_model()
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                model = model_data['model']
                label_encoder = model_data['label_encoder']
                all_symptoms = model_data['all_symptoms']
                symptom_to_index = model_data['symptom_to_index']
    else:
        print("No model found, training new one...")
        model, label_encoder, all_symptoms = train_model()
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            label_encoder = model_data['label_encoder']
            all_symptoms = model_data['all_symptoms']
            symptom_to_index = model_data['symptom_to_index']

    feature_metadata = {
        'feature_names': all_symptoms,
        'symptom_to_index': symptom_to_index,
        'label_encoder': label_encoder
    }

    return data_processor, single_symptom_mapping, model, feature_metadata


data_processor, single_symptom_mapping, model, feature_metadata = initialize_system()
st.session_state.data_processor = data_processor
st.session_state.single_symptom_mapping = single_symptom_mapping
st.session_state.model = model
st.session_state.feature_metadata = feature_metadata

all_symptoms = get_all_symptoms(feature_metadata)

if st.session_state.show_history:
    st.markdown("## Your Recommendation History")

    history = st.session_state.db.get_history()

    if history:
        for idx, record in enumerate(history):
            with st.expander(f"{record['created_at'].strftime('%Y-%m-%d %H:%M')} - {record['predicted_disease']} ({record['confidence']:.0%})"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Disease:** {record['predicted_disease']}")
                    st.markdown(f"**Confidence:** {record['confidence']:.0%}")
                    st.markdown(f"**Symptoms:** {', '.join(record['symptoms'])}")

                with col2:
                    st.markdown("**Medications:**")
                    for i, med in enumerate(record['medications'], 1):
                        st.markdown(f"{i}. {med}")

                    st.markdown("**Precautions:**")
                    for i, precaution in enumerate(record['precautions'], 1):
                        if precaution and precaution != "nan":
                            st.markdown(f"{i}. {precaution}")
    else:
        st.info("You don't have any recommendation history yet. Try getting some recommendations!")

elif not st.session_state.show_results:
    st.markdown("## Select Your Symptoms")

    selected_symptoms = st.multiselect(
        "Select your symptoms:",
        options=all_symptoms,
        default=st.session_state.selected_symptoms,
        key="symptom_selector"
    )

    st.session_state.selected_symptoms = selected_symptoms

    if selected_symptoms:
        st.markdown("### Rate the severity of your symptoms (1-5):")
        severity_dict = {}

        cols = st.columns(min(3, len(selected_symptoms)))

        for i, symptom in enumerate(selected_symptoms):
            col_idx = i % min(3, len(selected_symptoms))
            with cols[col_idx]:
                severity = st.slider(
                    f"{symptom}",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=f"severity_{symptom}"
                )
                severity_dict[symptom] = severity

        if st.button("Get Recommendations", type="primary", key="recommend_button"):
            if len(selected_symptoms) > 0:
                with st.spinner("Analyzing your symptoms..."):
                    if len(selected_symptoms) == 1 and st.session_state.single_symptom_mapping:
                        symptom = selected_symptoms[0]
                        if symptom in st.session_state.single_symptom_mapping:
                            top_diseases = st.session_state.single_symptom_mapping[symptom]

                            severity = severity_dict.get(symptom, 3)
                            severity_factor = 0.2 * severity + 0.8  # 1.0-1.8

                            disease, base_prob, _, _ = top_diseases[0]
                            confidence = min(base_prob * severity_factor * 2, 0.99)  # Scale up but cap at 0.99

                            medications = data_processor.get_medications_for_disease(disease)
                            precautions = data_processor.get_precautions_for_disease(disease)

                            alternatives = []
                            for i in range(1, min(6, len(top_diseases))):
                                alt_disease, alt_prob, _, _ = top_diseases[i]
                                alt_confidence = min(alt_prob * severity_factor * 2, 0.99)
                                alternatives.append((alt_disease, alt_confidence))

                            prediction_results = {
                                'predicted_disease': disease,
                                'confidence': confidence,
                                'medications': medications,
                                'precautions': precautions,
                                'alternative_diseases': alternatives
                            }

                            st.session_state.prediction_results = prediction_results
                            st.session_state.show_results = True
                        else:
                            symptom_dict = {symptom: severity_dict[symptom]}
                            input_features = process_user_input(symptom_dict, st.session_state.feature_metadata)
                            
                            probabilities = st.session_state.model.predict_proba(input_features)[0]
                            prediction_idx = np.argmax(probabilities)
                            confidence_scores = [probabilities[prediction_idx]]
                            
                            label_encoder = st.session_state.feature_metadata['label_encoder']
                            predicted_disease = label_encoder.inverse_transform([prediction_idx])[0]
                            confidence = confidence_scores[0]
                            
                            top_3_indices = np.argsort(probabilities)[-4:-1][::-1]
                            alternatives = []
                            for idx in top_3_indices:
                                if probabilities[idx] > 0.05:
                                    alt_disease = label_encoder.inverse_transform([idx])[0]
                                    alternatives.append((alt_disease, probabilities[idx]))

                            medications = data_processor.get_medications_for_disease(predicted_disease)
                            precautions = data_processor.get_precautions_for_disease(predicted_disease)

                            prediction_results = {
                                'predicted_disease': predicted_disease,
                                'confidence': confidence,
                                'medications': medications,
                                'precautions': precautions,
                                'alternative_diseases': alternatives
                            }

                            st.session_state.prediction_results = prediction_results
                            st.session_state.show_results = True
                    else:
                        symptom_dict = {symptom: severity_dict[symptom] for symptom in selected_symptoms}
                        input_features = process_user_input(symptom_dict, st.session_state.feature_metadata)
                        
                        probabilities = st.session_state.model.predict_proba(input_features)[0]
                        prediction_idx = np.argmax(probabilities)
                        confidence_scores = [probabilities[prediction_idx]]

                        label_encoder = st.session_state.feature_metadata['label_encoder']
                        predicted_disease = label_encoder.inverse_transform([prediction_idx])[0]
                        confidence = confidence_scores[0]
                        
                        top_3_indices = np.argsort(probabilities)[-4:-1][::-1]
                        alternatives = []
                        for idx in top_3_indices:
                            if probabilities[idx] > 0.05:
                                alt_disease = label_encoder.inverse_transform([idx])[0]
                                alternatives.append((alt_disease, probabilities[idx]))

                        medications = data_processor.get_medications_for_disease(predicted_disease)
                        precautions = data_processor.get_precautions_for_disease(predicted_disease)

                        prediction_results = {
                            'predicted_disease': predicted_disease,
                            'confidence': confidence,
                            'medications': medications,
                            'precautions': precautions,
                            'alternative_diseases': alternatives
                        }

                        st.session_state.prediction_results = prediction_results
                        st.session_state.show_results = True

                    if st.session_state.prediction_results:
                        results = st.session_state.prediction_results
                        try:
                            recommendation_id = st.session_state.db.save_recommendation(
                                selected_symptoms,
                                results['predicted_disease'],
                                results['confidence'],
                                results['medications'],
                                results['precautions'],
                                results['alternative_diseases']
                            )
                            st.session_state.recommendation_id = recommendation_id
                        except Exception as e:
                            st.error(f"Failed to save recommendation: {str(e)}")

            else:
                st.error("Please select at least one symptom.")
    else:
        st.info("Please select at least one symptom to get started.")

if st.session_state.show_results and st.session_state.prediction_results is not None:
    st.markdown("---")
    st.markdown("## Recommendation Results")

    results = st.session_state.prediction_results

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown("### Disease Prediction")
        disease = results['predicted_disease']
        confidence = results['confidence']

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': confidence * 100
                }
            }
        ))

        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### Predicted Disease: **{disease}**")
        if confidence < 0.5:
            st.warning(" Low confidence prediction. Consider consulting a healthcare professional.")
        elif confidence < 0.8:
            st.info("ℹ️ Moderate confidence prediction. Verify symptoms or add more if possible.")
        else:
            st.success("✅ High confidence prediction.")

    with col2:
        st.markdown("### Recommended Medications")
        medications = results['medications']

        for i, med in enumerate(medications, 1):
            st.markdown(f"**{i}.** {med}")

        if medications:
            st.download_button(
                label="Download Medication List",
                data="\n".join([f"{i}. {med}" for i, med in enumerate(medications, 1)]),
                file_name="recommended_medications.txt",
                mime="text/plain"
            )

    with col3:
        st.markdown("### Precautions")
        precautions = results['precautions']

        for i, precaution in enumerate(precautions, 1):
            if precaution and precaution != "nan":
                st.markdown(f"**{i}.** {precaution}")

    st.markdown("### Alternative Possible Conditions")

    alternatives = results['alternative_diseases']

    if alternatives:
        alt_df = pd.DataFrame(alternatives, columns=['Disease', 'Probability'])

        alt_df = alt_df.sort_values('Probability', ascending=False)

        fig = px.bar(
            alt_df,
            x='Probability',
            y='Disease',
            orientation='h',
            color='Probability',
            color_continuous_scale='Blues',
            labels={'Probability': 'Confidence Score', 'Disease': 'Alternative Condition'}
        )

        fig.update_layout(
            height=min(350, 100 + len(alternatives) * 30),
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis={'categoryorder': 'total ascending'}
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info("These are alternative conditions that might match your symptoms. The higher the confidence score, the more likely the condition.")
    else:
        st.info("No alternative conditions found.")

    if len(st.session_state.selected_symptoms) == 1:
        st.success("✅ This prediction was made using direct symptom-to-disease mapping for maximum accuracy with single symptoms.")
    else:
        st.info(f"ℹ️ This prediction was made using machine learning analysis of {len(st.session_state.selected_symptoms)} symptoms.")

    st.markdown("---")
    st.markdown("## Feedback")
    st.write("Was this recommendation helpful? Your feedback helps us improve.")

    col1, col2 = st.columns([1, 2])

    with col1:
        is_accurate = st.radio(
            "Was the recommendation accurate?",
            options=["Yes", "No", "Not sure"],
            index=2,
            key="feedback_accuracy"
        )

    with col2:
        feedback_text = st.text_area(
            "Additional comments (optional):",
            key="feedback_text",
            max_chars=500
        )

    if st.button("Submit Feedback", key="submit_feedback"):
        accuracy_map = {"Yes": True, "No": False, "Not sure": None}
        accuracy_bool = accuracy_map[is_accurate]

        if st.session_state.recommendation_id:
            try:
                feedback_id = st.session_state.db.save_feedback(
                    st.session_state.recommendation_id,
                    accuracy_bool,
                    feedback_text if feedback_text else None
                )
                if feedback_id:
                    st.success("Thank you for your feedback!")
                else:
                    st.error("Failed to save feedback. Please try again.")
            except Exception as e:
                st.error(f"Error saving feedback: {str(e)}")
        else:
            st.success("Thank you for your feedback!")

        import time
        time.sleep(1)

        st.session_state.selected_symptoms = []
        st.session_state.show_results = False
        st.session_state.prediction_results = None

    if st.button("Start Over (without feedback)", key="start_over"):
        st.session_state.selected_symptoms = []
        st.session_state.show_results = False
        st.session_state.prediction_results = None
