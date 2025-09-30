# ====================================================
# DS Automation Assignment - Churn Prediction with PyCaret
# ====================================================

import pandas as pd
from pycaret.classification import load_model, predict_model


# ====================================================
# Step 1: Preprocessing Function for New Data
# ====================================================
def preprocess_new_data(df):
    """
    Recreates the engineered features and ensures the 
    input dataframe has all required columns for prediction.
    """
    # Recreate engineered features
    df['ChargesPerMonth_from_total'] = df['TotalCharges'] / (df['tenure'] + 1e-5)
    df['MonthsOfService_est'] = df['tenure']

    # One-hot encode selected categoricals
    df = pd.get_dummies(df, columns=['PhoneService', 'Contract', 'PaymentMethod'])

    # Ensure expected columns are present
    expected_cols = [
        'ChargesPerMonth_from_total', 'MonthsOfService_est',
        'PhoneService_Yes', 'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    
    return df


# ====================================================
# Step 2: Prediction Function
# ====================================================
def predict_churn(new_data_path, model_path="final_churn_model"):
    """
    Loads the trained PyCaret model and applies it to new churn data.
    Saves predictions to CSV.
    """
    # Load new churn test data
    new_data = pd.read_csv(new_data_path)

    # Preprocess
    df_processed = preprocess_new_data(new_data)

    # Load model
    model = load_model(model_path)

    # Predict churn
    predictions = predict_model(model, data=df_processed)

    # Merge customer IDs with predictions
    results = new_data.copy()
    if 'prediction_label' in predictions.columns:
        results['prediction_label'] = predictions['prediction_label']
    if 'prediction_score' in predictions.columns:
        results['prediction_score'] = predictions['prediction_score']

    # Save predictions
    results.to_csv("new_churn_predictions.csv", index=False)
    print("✅ Predictions saved to new_churn_predictions.csv")

    return results


# ====================================================
# Step 3: Main Execution
# ====================================================
if __name__ == "__main__":
    # Path to new data file
    new_data_file = "new_churn_data.csv"

    # Run prediction
    results = predict_churn(new_data_file)

    # Print preview
    print("Predictions on new data:")
    print(results[['customerID', 'prediction_label']].head())
