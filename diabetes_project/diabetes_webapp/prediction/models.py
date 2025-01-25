import joblib
import numpy as np

# Load models
lasso_model = joblib.load('../../diabetes_project/models/lasso_model.pkl')
rf_model = joblib.load('../../diabetes_project/models/rf_model.pkl')
xgb_model = joblib.load('../../diabetes_project/models/xgb_model.pkl')
supermodel_weights = joblib.load('../../diabetes_project/models/supermodel_weights.pkl')
supermodel_weights = {'lasso': 0.1, 'random_forest': 0.4, 'xgboost': 0.5}

# Full list of feature names used during training
feature_names = [
    'Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure',
    'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10',
    'Feature11', 'Feature12', 'Feature13', 'Feature14', 'Feature15',
    'Feature16', 'Feature17', 'Feature18', 'Feature19', 'Feature20',
    'Feature21', 'Feature22', 'Feature23', 'Feature24'
]

def predict_diabetes(data):
    # Initialize transformed input with default values (zeros)
    transformed_input = np.zeros(len(feature_names))

    # Map provided features to the transformed input
    for i, feature in enumerate(['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure']):
        transformed_input[i] = data[i]

    # Reshape input to match model requirements
    transformed_input = transformed_input.reshape(1, -1)

    # Get predictions
    lasso_pred = lasso_model.predict(transformed_input)[0]

    # Normalize LASSO prediction
    lasso_min = 0  # Replace with the actual minimum value of LASSO outputs
    lasso_max = 5  # Replace with the actual maximum value of LASSO outputs
    lasso_pred_normalized = (lasso_pred - lasso_min) / (lasso_max - lasso_min)

    rf_prob = rf_model.predict_proba(transformed_input)[0][1]
    xgb_prob = xgb_model.predict_proba(transformed_input)[0][1]

    # Weighted average for supermodel
    final_prob = (
        supermodel_weights['lasso'] * lasso_pred_normalized +
        supermodel_weights['random_forest'] * rf_prob +
        supermodel_weights['xgboost'] * xgb_prob
    )

    print("LASSO Prediction:", lasso_pred,lasso_pred_normalized)
    print("Random Forest Probability:", rf_prob)
    print("XGBoost Probability:", xgb_prob)

    return "Diabetic" if final_prob > 0.5 else "Non-Diabetic"