# main.py

from diabetes_project.preprocessing.preprocess import load_and_preprocess_data
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

def train_and_save_models(X_train, y_train):
    # Ensure data is in the correct format
    X_train = np.array(X_train)  # Convert to NumPy array if not already
    y_train = np.array(y_train)  # Convert to NumPy array if not already

    # Handle missing values in X_train
    X_train = np.nan_to_num(X_train)

    # Train LASSO Regression
    try:
        lasso_model = Lasso(alpha=0.1)
        lasso_model.fit(X_train, y_train)
        joblib.dump(lasso_model, 'diabetes_project/models/lasso_model.pkl')
        print("LASSO model trained and saved successfully.")
    except Exception as e:
        print("Error training LASSO model:", e)

    # Train Random Forest
    try:
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        joblib.dump(rf_model, 'diabetes_project/models/rf_model.pkl')
        print("Random Forest model trained and saved successfully.")
    except Exception as e:
        print("Error training Random Forest model:", e)

    # Train XGBoost
    try:
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_model.fit(X_train, y_train)
        joblib.dump(xgb_model, 'diabetes_project/models/xgb_model.pkl')
        print("XGBoost model trained and saved successfully.")
    except Exception as e:
        print("Error training XGBoost model:", e)

    # Save model weights for ensemble
    try:
        supermodel_weights = {'lasso': 0.3, 'random_forest': 0.4, 'xgboost': 0.3}
        joblib.dump(supermodel_weights, 'diabetes_project/models/supermodel_weights.pkl')
        print("Supermodel weights saved successfully.")
    except Exception as e:
        print("Error saving supermodel weights:", e)

def evaluate_models(X_test, y_test):
    # Load models
    lasso_model = joblib.load('diabetes_project/models/lasso_model.pkl')
    rf_model = joblib.load('diabetes_project/models/rf_model.pkl')
    xgb_model = joblib.load('diabetes_project/models/xgb_model.pkl')
    supermodel_weights = joblib.load('diabetes_project/models/supermodel_weights.pkl')

    # Ensure X_test is a NumPy array
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Handle missing values in X_test (if any)
    X_test = np.nan_to_num(X_test)

    # Get predictions
    try:
        lasso_preds = lasso_model.predict(X_test)
    except Exception as e:
        print("Error in LASSO prediction:", e)
        return

    try:
        rf_probs = rf_model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print("Error in Random Forest prediction:", e)
        return

    try:
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print("Error in XGBoost prediction:", e)
        return

    # Weighted average for supermodel
    final_probs = (supermodel_weights['lasso'] * lasso_preds +
                   supermodel_weights['random_forest'] * rf_probs +
                   supermodel_weights['xgboost'] * xgb_probs)
    final_preds = (final_probs > 0.5).astype(int)

    # Evaluate
    print("Supermodel Accuracy:", accuracy_score(y_test, final_preds))
    print(classification_report(y_test, final_preds))
    
if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Train and save models
    train_and_save_models(X_train, y_train)

    # Evaluate supermodel
    evaluate_models(X_test, y_test)
