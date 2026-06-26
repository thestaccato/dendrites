import pandas as pd
import numpy as np
from ast import literal_eval
import re

class DataProcessor:
    """
    Handles data loading, preprocessing, and feature extraction
    for the medication recommendation system.
    """
    
    def __init__(self, dataset_path):
        """
        Initialize the data processor with the dataset
        
        Args:
            dataset_path: Path to the dataset CSV file
        """
        self.dataset_path = dataset_path
        self.df = self._load_and_preprocess_data()
        self.all_symptoms = self._extract_all_symptoms()
        self.symptom_to_disease_map = self._create_symptom_to_disease_map()
        self.symptom_to_disease_freq = self._create_symptom_to_disease_frequency()
        self.disease_to_medication_map = self._create_disease_to_medication_map()
        self.disease_to_precaution_map = self._create_disease_to_precaution_map()
        
    def _load_and_preprocess_data(self):
        """
        Load and preprocess the dataset
        
        Returns:
            Preprocessed DataFrame
        """
        df = pd.read_csv(self.dataset_path)
        
        list_columns = ['Medication', 'All_Symptoms', 'Symptoms_with_Severity']
        for col in list_columns:
            df[col] = df[col].apply(self._safe_eval)
        
        return df
    
    def _safe_eval(self, val):
        """
        Safely evaluate string representations of lists
        
        Args:
            val: String representation of a list
            
        Returns:
            Python list object or empty list if evaluation fails
        """
        if isinstance(val, list):
            return val
        
        if pd.isna(val) or val == '':
            return []
            
        try:
            val = val.replace("'", "\"")
            val = re.sub(r'nan', 'null', val)
            return literal_eval(val)
        except (ValueError, SyntaxError):
            try:
                pattern = r'\(([^,]+),\s*([^)]+)\)'
                matches = re.findall(pattern, val)
                if matches:
                    return [(item[0].strip(), 
                             int(item[1]) if item[1].strip().isdigit() else None) 
                            for item in matches]
                return []
            except:
                return []
    
    def _extract_all_symptoms(self):
        """
        Extract all unique symptoms from the dataset
        
        Returns:
            List of all unique symptoms
        """
        all_symptoms = set()
        
        for symptoms_list in self.df['All_Symptoms']:
            if isinstance(symptoms_list, list):
                for symptom in symptoms_list:
                    if symptom and not pd.isna(symptom):
                        symptom = symptom.strip()
                        all_symptoms.add(symptom)
        
        return sorted(list(all_symptoms))
    
    def _create_symptom_to_disease_map(self):
        """
        Create a mapping from each symptom to the diseases it's associated with
        
        Returns:
            Dictionary mapping symptoms to diseases
        """
        symptom_to_disease = {}
        
        for _, row in self.df.iterrows():
            disease = row['Disease']
            symptoms = row['All_Symptoms']
            
            if isinstance(symptoms, list):
                for symptom in symptoms:
                    if symptom and not pd.isna(symptom):
                        symptom = symptom.strip()
                        if symptom not in symptom_to_disease:
                            symptom_to_disease[symptom] = []
                        if disease not in symptom_to_disease[symptom]:
                            symptom_to_disease[symptom].append(disease)
        
        return symptom_to_disease
    
    def _create_symptom_to_disease_frequency(self):
        """
        Create a mapping from each symptom to its frequency in each disease
        
        Returns:
            Dictionary mapping symptoms to frequency in each disease
        """
        symptom_to_disease_freq = {}
        
        disease_symptom_count = {}
        for _, row in self.df.iterrows():
            disease = row['Disease']
            symptoms = row['All_Symptoms']
            
            if isinstance(symptoms, list):
                if disease not in disease_symptom_count:
                    disease_symptom_count[disease] = {}
                    
                for symptom in symptoms:
                    if symptom and not pd.isna(symptom):
                        symptom = symptom.strip()
                        if symptom not in disease_symptom_count[disease]:
                            disease_symptom_count[disease][symptom] = 0
                        disease_symptom_count[disease][symptom] += 1
        
        for disease, symptom_counts in disease_symptom_count.items():
            total_entries = sum(symptom_counts.values())
            
            for symptom, count in symptom_counts.items():
                if symptom not in symptom_to_disease_freq:
                    symptom_to_disease_freq[symptom] = {}
                
                symptom_to_disease_freq[symptom][disease] = {
                    'frequency': count / total_entries,
                    'count': count
                }
        
        return symptom_to_disease_freq
    
    def _create_disease_to_medication_map(self):
        """
        Create a mapping from each disease to its recommended medications
        
        Returns:
            Dictionary mapping diseases to medications
        """
        disease_to_medication = {}
        
        fallback_medications = {
            'Peptic ulcer diseae': ['Omeprazole', 'Amoxicillin', 'Clarithromycin', 'Metronidazole', 'Pantoprazole'],
            'Diabetes ': ['Insulin', 'Metformin', 'Glipizide', 'Sitagliptin', 'Empagliflozin'],
            'Hypertension ': ['Lisinopril', 'Amlodipine', 'Hydrochlorothiazide', 'Losartan', 'Metoprolol'],
            '(vertigo) Paroymsal  Positional Vertigo': ['Betahistine', 'Meclizine', 'Dimenhydrinate', 'Diazepam', 'Epley Maneuver (Procedure)'],
            'Hypothyroidism': ['Levothyroxine', 'Liothyronine'],
            'Hyperthyroidism': ['Methimazole', 'Propylthiouracil', 'Radioactive Iodine'],
            'Dimorphic hemmorhoids(piles)': ['Hydrocortisone cream', 'Witch hazel', 'Psyllium husk', 'Sitz bath (Procedure)']
        }
        
        for _, row in self.df.iterrows():
            disease = row['Disease']
            medications = row['Medication']
            
            if disease not in disease_to_medication:
                disease_to_medication[disease] = []
                
            if isinstance(medications, list) and len(medications) > 0:
                for med in medications:
                    if med and not pd.isna(med) and med not in disease_to_medication[disease]:
                        disease_to_medication[disease].append(med)
            
            if not disease_to_medication[disease] and disease in fallback_medications:
                 disease_to_medication[disease] = fallback_medications[disease]
        
        for disease, meds in fallback_medications.items():
            if disease in disease_to_medication and not disease_to_medication[disease]:
                disease_to_medication[disease] = meds
                
        return disease_to_medication
    
    def _create_disease_to_precaution_map(self):
        """
        Create a mapping from each disease to its recommended precautions
        
        Returns:
            Dictionary mapping diseases to precautions
        """
        disease_to_precaution = {}
        
        precaution_columns = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
        
        for _, row in self.df.iterrows():
            disease = row['Disease']
            
            if disease not in disease_to_precaution:
                disease_to_precaution[disease] = []
                
            for col in precaution_columns:
                precaution = row[col]
                if precaution and not pd.isna(precaution) and precaution not in disease_to_precaution[disease]:
                    disease_to_precaution[disease].append(precaution)
        
        return disease_to_precaution
    
    def get_all_symptoms(self):
        """
        Get the list of all unique symptoms
        
        Returns:
            List of all unique symptoms
        """
        return self.all_symptoms
    
    def get_disease_from_symptom(self, symptom):
        """
        Get diseases associated with a specific symptom
        
        Args:
            symptom: The symptom to look up
            
        Returns:
            List of diseases associated with the symptom
        """
        if symptom in self.symptom_to_disease_map:
            return self.symptom_to_disease_map[symptom]
        return []
    
    def get_disease_frequency_for_symptom(self, symptom):
        """
        Get the frequency of a symptom in different diseases
        
        Args:
            symptom: The symptom to look up
            
        Returns:
            Dictionary mapping diseases to frequency information
        """
        if symptom in self.symptom_to_disease_freq:
            return self.symptom_to_disease_freq[symptom]
        return {}
    
    def get_medications_for_disease(self, disease):
        """
        Get medications recommended for a specific disease
        
        Args:
            disease: The disease to look up
            
        Returns:
            List of medications for the disease
        """
        if disease in self.disease_to_medication_map:
            return self.disease_to_medication_map[disease]
        return []
    
    def get_precautions_for_disease(self, disease):
        """
        Get precautions recommended for a specific disease
        
        Args:
            disease: The disease to look up
            
        Returns:
            List of precautions for the disease
        """
        if disease in self.disease_to_precaution_map:
            return self.disease_to_precaution_map[disease]
        return []
    
    def create_symptoms_vector(self, selected_symptoms):
        """
        Create a binary feature vector from selected symptoms
        
        Args:
            selected_symptoms: List of selected symptoms
            
        Returns:
            Binary feature vector representing the symptoms
        """
        vector = np.zeros(len(self.all_symptoms))
        
        for symptom in selected_symptoms:
            if symptom in self.all_symptoms:
                index = self.all_symptoms.index(symptom)
                vector[index] = 1
        
        return vector
    
    def get_diseases(self):
        """
        Get the list of all unique diseases
        
        Returns:
            List of all unique diseases
        """
        return self.df['Disease'].unique().tolist()
    
    def get_dataset(self):
        """
        Get the preprocessed dataset
        
        Returns:
            Preprocessed DataFrame
        """
        return self.df
