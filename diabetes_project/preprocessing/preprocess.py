# preprocessing/preprocess.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data():
    # Load UCI dataset
    uci_data = pd.read_excel('diabetes_project/datasets/uci_combined.xlsx')
    # Map 'Code' to meaningful feature names
    code_mapping = {
        "Pre-breakfast blood glucose measurement": "PreBreakfastGlucose",
        "Regular insulin dose": "RegularInsulin",
        "NPH insulin dose": "NPHInsulin",
        "Pre-supper blood glucose measurement": "PreSupperGlucose"
    }
    uci_data['Code'] = uci_data['Code'].map(code_mapping)
    # Pivot to create feature columns
    uci_pivot = uci_data.pivot_table(index=['Date', 'Time'], columns='Code', values='Value', aggfunc='first').reset_index()
    uci_pivot.fillna(0, inplace=True)  # Fill missing values with 0

    # Load Kaggle dataset
    kaggle_data = pd.read_csv('diabetes_project/datasets/diabetes.csv')

    # Load Dataset1_PIDD (Mendeley)
    pidd_data = pd.read_excel('diabetes_project/datasets/Dataset1_PIDD.xlsx')
    pidd_data.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                         'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

    # Load Dataset2_ESDRPD (Mendeley)
    esdrpd_data = pd.read_excel('diabetes_project/datasets/Dataset2_ESDRPD.xlsx')
    encoder = LabelEncoder()
    for col in esdrpd_data.select_dtypes(include='object').columns:
        esdrpd_data[col] = encoder.fit_transform(esdrpd_data[col])

    # Normalize numeric features in Kaggle and PIDD diabetes_project/datasets
    scaler = StandardScaler()
    numeric_features = ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure']
    kaggle_data[numeric_features] = scaler.fit_transform(kaggle_data[numeric_features])
    pidd_data[numeric_features] = scaler.fit_transform(pidd_data[numeric_features])

    # Combine Kaggle and PIDD diabetes_project/datasets
    combined_data = pd.concat([kaggle_data, pidd_data], ignore_index=True)

    # Add ESDrpd data (excluding class label for features)
    esdrpd_features = esdrpd_data.drop(columns=['Class'])
    combined_data = pd.concat([combined_data, esdrpd_features], axis=1)

    # Handle missing values in combined dataset
    combined_data.fillna(combined_data.mean(), inplace=True)

    # Feature-target split
    X = combined_data.drop(columns=['Outcome', 'Class'], errors='ignore')
    y = combined_data['Outcome'] if 'Outcome' in combined_data.columns else combined_data['Class']

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
