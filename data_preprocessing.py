import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load the dataset from the csv file
    
    Args:
        file_path: Path to the dataset
    
    Returns:
        DataFrame: Loaded dataset
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """
    Clean the dataset by handling missing values, parsing lists, etc.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame: Cleaned DataFrame
    """
    data = df.copy()
    
    for col in ['All_Symptoms', 'Symptoms_with_Severity', 'Medication']:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: convert_to_list(x))
    
    data['All_Symptoms'] = data['All_Symptoms'].apply(lambda x: [] if pd.isna(x).any() else x)
    if 'Symptoms_with_Severity' in data.columns:
        data['Symptoms_with_Severity'] = data['Symptoms_with_Severity'].apply(lambda x: [] if pd.isna(x).any() else x)
    
    for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
        if col in data.columns:
            data[col] = data[col].fillna('')
    
    for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
        if col in data.columns:
            data[col] = data[col].fillna('')
    
    return data

def convert_to_list(value):
    """
    Convert a string representation of a list to an actual list
    
    Args:
        value: String or list to convert
    
    Returns:
        list: Converted list
    """
    if pd.isna(value):
        return []
    
    if isinstance(value, list):
        return value
    
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [value]

def extract_features(df):
    """
    Extract and process features from the dataset
    
    Args:
        df: Input DataFrame
    
    Returns:
        tuple: X features, y labels, and feature metadata
    """
    all_symptoms = []
    for symptoms_list in df['All_Symptoms']:
        if isinstance(symptoms_list, list):
            all_symptoms.extend([s.strip() if isinstance(s, str) else s for s in symptoms_list if not pd.isna(s)])
    
    unique_symptoms = sorted(list(set([s for s in all_symptoms if isinstance(s, str)])))
    
    symptom_to_index = {symptom: i for i, symptom in enumerate(unique_symptoms)}
    
    def process_symptoms_with_severity(symptoms_with_severity):
        result = {}
        
        if not symptoms_with_severity or not isinstance(symptoms_with_severity, list):
            return result
            
        for item in symptoms_with_severity:
            if isinstance(item, tuple) and len(item) == 2:
                symptom, severity = item
                if isinstance(symptom, str):
                    symptom = symptom.strip()
                    
                if severity is None or (isinstance(severity, float) and np.isnan(severity)):
                    severity = 3
                    
                result[symptom] = severity
                
        return result
    
    X_features = []
    y_labels = []
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['Disease'])
    
    for i, row in df.iterrows():
        symptoms_list = row['All_Symptoms'] if isinstance(row['All_Symptoms'], list) else []
        
        symptoms_with_severity_dict = {}
        if 'Symptoms_with_Severity' in df.columns:
            symptoms_with_severity = row['Symptoms_with_Severity'] if isinstance(row['Symptoms_with_Severity'], list) else []
            symptoms_with_severity_dict = process_symptoms_with_severity(symptoms_with_severity)
        
        features = np.zeros(len(unique_symptoms))
        
        for symptom in symptoms_list:
            if isinstance(symptom, str) and symptom.strip() in symptom_to_index:
                symptom_idx = symptom_to_index[symptom.strip()]
                
                severity = symptoms_with_severity_dict.get(symptom.strip(), 3)
                features[symptom_idx] = float(severity) / 5.0
        
        X_features.append(features)
        y_labels.append(y_encoded[i])
    
    X_features = np.array(X_features)
    y_labels = np.array(y_labels)
    
    feature_metadata = {
        'feature_names': unique_symptoms,
        'symptom_to_index': symptom_to_index,
        'label_encoder': label_encoder
    }
    
    return X_features, y_labels, feature_metadata

def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets
    
    Args:
        X: Input features
        y: Target labels
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
    
    Returns:
        tuple: Training and testing data splits
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test



def preprocess_pipeline(file_path, sequence_length=5):
    """
    Complete preprocessing pipeline for the dataset
    
    Args:
        file_path: Path to the dataset
        sequence_length: Length of sequences for LSTM
    
    Returns:
        tuple: Processed data and metadata
    """
    # Load data
    raw_data = load_data(file_path)
    if raw_data is None:
        return None, None, None, None, None, None, None, None
    
    clean_df = clean_data(raw_data)
    
    X, y, feature_metadata = extract_features(clean_df)
    
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    if len(X_train) < 500:
        print(f"Small dataset detected ({len(X_train)} samples). Performing augmentation...")
        
        X_aug = []
        y_aug = []
        
        for i in range(len(X_train)):
            X_aug.append(X_train[i])
            y_aug.append(y_train[i])
            
            non_zeros = np.nonzero(X_train[i])[0]
            if len(non_zeros) > 1:
                x_new = X_train[i].copy()
                drop_idx = np.random.choice(non_zeros)
                x_new[drop_idx] = 0
                X_aug.append(x_new)
                y_aug.append(y_train[i])
                
        X_train = np.array(X_aug)
        y_train = np.array(y_aug)
        print(f"Augmentation complete. New training size: {len(X_train)}")

    return X_train, X_test, y_train, y_test, feature_metadata, clean_df

def process_user_input(symptoms, feature_metadata, sequence_length=5):
    """
    Process user input symptoms for prediction
    
    Args:
        symptoms: Dictionary mapping symptoms to severity values
        feature_metadata: Metadata about features
        sequence_length: Length of sequences for LSTM
    
    Returns:
        tuple: Processed input for model prediction (sequence features, flat features)
    """
    symptom_to_index = feature_metadata['symptom_to_index']
    
    num_features = len(feature_metadata['feature_names'])
    
    features = np.zeros(num_features)
    
    for symptom, severity in symptoms.items():
        if symptom in symptom_to_index:
            symptom_idx = symptom_to_index[symptom]
            features[symptom_idx] = float(severity) / 5.0
    
    X_flat = features.reshape(1, -1)
    
    return X_flat

def get_all_symptoms(feature_metadata):
    """
    Get a list of all symptoms for user selection
    
    Args:
        feature_metadata: Metadata about features
    
    Returns:
        list: All available symptoms
    """
    return feature_metadata['feature_names']
