import os
import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import preprocess_pipeline
from utils import get_model_path

warnings.filterwarnings("ignore")

def train_model():
    """
    Train a Random Forest model on the dataset.
    
    Returns:
        Trained model, label_encoder, all_symptoms
    """
    print("Loading and preprocessing data...")
    
    original_dataset = 'attached_assets/merged_health_dataset.csv'
    augmented_dataset = 'attached_assets/augmented_dataset.csv'
    
    dataset_path = augmented_dataset if os.path.exists(augmented_dataset) else original_dataset
    print(f"Using dataset: {dataset_path}")
    
    X_train, X_test, y_train, y_test, feature_metadata, clean_df = preprocess_pipeline(dataset_path)
    
    if X_train is None:
        print("Failed to load or preprocess data.")
        return None, None, None
        
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    rf = RandomForestClassifier(
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1
    )
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    }
    
    print("Tuning hyperparameters...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("------------------------------------------------")
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=feature_metadata['label_encoder'].classes_))
    print("------------------------------------------------")
    
    model_data = {
        'model': best_model,
        'label_encoder': feature_metadata['label_encoder'],
        'all_symptoms': feature_metadata['feature_names'],
        'symptom_to_index': feature_metadata['symptom_to_index']
    }
    
    os.makedirs(os.path.dirname(get_model_path()), exist_ok=True)
    
    with open(get_model_path(), 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {get_model_path()}")
    
    return best_model, feature_metadata['label_encoder'], feature_metadata['feature_names']

if __name__ == "__main__":
    try:
        print("Training a new model...")
        model, label_encoder, all_symptoms = train_model()
            
        print("Model training complete.")
    except Exception as e:
        print(f"Error during model training: {str(e)}")
