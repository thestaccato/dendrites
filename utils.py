import os
import pickle
import numpy as np

def get_model_path():
    """
    Get the path to save/load the model
    
    Returns:
        Path to the model file
    """
    model_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    return os.path.join(model_dir, 'med_recommendation_model.pkl')

def load_model(model_path):
    """
    Load the trained model from disk
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model and label encoder
    """
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def symptoms_to_vector(symptoms, all_symptoms):
    """
    Convert a list of symptoms to a binary feature vector
    
    Args:
        symptoms: List of symptoms
        all_symptoms: List of all possible symptoms
        
    Returns:
        Binary feature vector
    """
    vector = np.zeros(len(all_symptoms))
    
    for symptom in symptoms:
        if symptom in all_symptoms:
            index = all_symptoms.index(symptom)
            vector[index] = 1
    
    return vector
