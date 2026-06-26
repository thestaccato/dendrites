import pandas as pd
import pickle
import os
from collections import defaultdict

def generate_single_symptom_mapping(df=None, dataset_path=None, output_path='models/single_symptom_mapping.pkl'):
    if df is None:
        if dataset_path is None:
            raise ValueError("Either df or dataset_path must be provided")
        df = pd.read_csv(dataset_path)

    print("Generating single symptom to disease mapping...")

    symptom_disease_count = defaultdict(lambda: defaultdict(int))
    symptom_count = defaultdict(int)
    disease_count = defaultdict(int)
    total_rows = len(df)

    for _, row in df.iterrows():
        disease = row['Disease']
        disease_count[disease] += 1

        for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
            symptom = row[col]
            if pd.notna(symptom) and symptom.strip():
                symptom = symptom.strip()
                symptom_count[symptom] += 1
                symptom_disease_count[symptom][disease] += 1

    symptom_to_disease_prob = {}
    for symptom, disease_counts in symptom_disease_count.items():
        symptom_to_disease_prob[symptom] = {}
        total_symptom_occurrences = sum(disease_counts.values())

        for disease, count in disease_counts.items():
            prior_disease_prob = disease_count[disease] / total_rows
            likelihood = count / total_symptom_occurrences
            posterior_prob = likelihood * prior_disease_prob

            symptom_to_disease_prob[symptom][disease] = {
                'probability': posterior_prob,
                'count': count,
                'total_symptom_occurrences': total_symptom_occurrences
            }

    sorted_mapping = {}
    for symptom, disease_probs in symptom_to_disease_prob.items():
        sorted_mapping[symptom] = sorted(
            [(disease, info['probability'], info['count'], info['total_symptom_occurrences'])
             for disease, info in disease_probs.items()],
            key=lambda x: x[1],
            reverse=True
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(sorted_mapping, f)

    print(f"Single symptom mapping saved to {output_path}")
    return sorted_mapping


def load_symptom_mapping(mapping_path='models/single_symptom_mapping.pkl'):
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'rb') as f:
                mapping = pickle.load(f)
            return mapping
        except Exception as e:
            print(f"Error loading symptom mapping: {e}")
            return None
    else:
        print(f"Symptom mapping file not found at {mapping_path}")
        return None
