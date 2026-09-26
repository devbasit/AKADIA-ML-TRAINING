from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Literal

app = FastAPI(title="ML Model Deployment API")

# Load models and scaler
# In a real application, you might want to load these once at startup
# and store them globally or in app.state.
regression_models = {
    'linear_regression': joblib.load('linear_regression_model.pkl'),
    'decision_tree_regressor': joblib.load('decision_tree_regressor_model.pkl')
}
classification_models = {
    'logistic_regression': joblib.load('logistic_regression_classifier_model.pkl'),
    'decision_tree_classifier': joblib.load('decision_tree_classifier_model.pkl')
}
clustering_models = {
    'kmeans': joblib.load('kmeans_clustering_model.pkl'),
    'agglomerative_clustering': joblib.load('agglomerative_clustering_model.pkl')
}
scaler = joblib.load('scaler.pkl')

# Pydantic models for request bodies
class RegressionInput(BaseModel):
    study_hours: float
    attendance_rate: float
    model_name: Literal["linear_regression", "decision_tree_regressor"]

class ClassificationInput(BaseModel):
    study_hours: float
    attendance_rate: float
    gender: Literal["Male", "Female"]
    internet_access: Literal["Yes", "No"]
    model_name: Literal["logistic_regression", "decision_tree_classifier"]

class ClusteringInput(BaseModel):
    study_hours: float
    attendance_rate: float
    exam_score: float
    model_name: Literal["kmeans", "agglomerative_clustering"]

@app.post("/predict/regression")
async def predict_regression(input_data: RegressionInput):
    model = regression_models.get(input_data.model_name)
    if not model:
        raise HTTPException(status_code=400, detail="Invalid regression model name")
    
    df = pd.DataFrame([{
        'Study_Hours': input_data.study_hours,
        'Attendance_Rate': input_data.attendance_rate
    }])
    
    prediction = model.predict(df)[0]
    return {"task": "regression", "model": input_data.model_name, "predicted_exam_score": round(prediction, 2)}

@app.post("/predict/classification")
async def predict_classification(input_data: ClassificationInput):
    model = classification_models.get(input_data.model_name)
    if not model:
        raise HTTPException(status_code=400, detail="Invalid classification model name")

    # Prepare input data for one-hot encoding
    data_dict = {
        'Study_Hours': [input_data.study_hours],
        'Attendance_Rate': [input_data.attendance_rate],
        'Gender': [input_data.gender],
        'Internet_Access': [input_data.internet_access]
    }
    df = pd.DataFrame(data_dict)

    # Apply one-hot encoding consistent with training
    # Assuming columns were 'Gender_Male', 'Internet_Access_Yes'
    df['Gender_Male'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['Internet_Access_Yes'] = df['Internet_Access'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Select and order features as trained
    feature_cols = ['Study_Hours', 'Attendance_Rate', 'Gender_Male', 'Internet_Access_Yes']
    input_features = df[feature_cols]
    
    prediction = model.predict(input_features)[0]
    result = "Pass" if prediction == 1 else "Fail"
    return {"task": "classification", "model": input_data.model_name, "predicted_pass_fail": result}

@app.post("/predict/clustering")
async def predict_clustering(input_data: ClusteringInput):
    model = clustering_models.get(input_data.model_name)
    if not model:
        raise HTTPException(status_code=400, detail="Invalid clustering model name")
    
    df = pd.DataFrame([{
        'Study_Hours': input_data.study_hours,
        'Attendance_Rate': input_data.attendance_rate,
        'Exam_Score': input_data.exam_score
    }])

    # Scale the input data
    scaled_input = scaler.transform(df[['Study_Hours', 'Attendance_Rate', 'Exam_Score']])
    
    cluster_assignment = model.predict(scaled_input)[0]
    return {"task": "clustering", "model": input_data.model_name, "assigned_cluster": int(cluster_assignment)}
