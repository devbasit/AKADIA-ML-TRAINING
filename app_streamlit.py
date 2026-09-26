import streamlit as st
import joblib
import pandas as pd

# Load models and scaler
@st.cache_resource
def load_models():
    models = {
        'regression': {
            'Linear Regression': joblib.load('linear_regression_model.pkl'),
            'Decision Tree Regressor': joblib.load('decision_tree_regressor_model.pkl')
        },
        'classification': {
            'Logistic Regression': joblib.load('logistic_regression_classifier_model.pkl'),
            'Decision Tree Classifier': joblib.load('decision_tree_classifier_model.pkl')
        },
        'clustering': {
            'K-Means': joblib.load('kmeans_clustering_model.pkl'),
            'Agglomerative Clustering': joblib.load('agglomerative_clustering_model.pkl')
        }
    }
    scaler = joblib.load('scaler.pkl')
    return models, scaler

models, scaler = load_models()

st.title("ML Model Deployment Test")

# Task Selection
task = st.sidebar.selectbox(
    "Select Task",
    ("Regression", "Classification", "Clustering")
)

# Input Data
st.header("Input Data")

if task == "Regression":
    model_choice = st.sidebar.selectbox(
        "Select Regression Model",
        list(models['regression'].keys())
    )
    study_hours = st.number_input("Study Hours", min_value=0.0, max_value=20.0, value=5.0)
    attendance_rate = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=80.0)
    input_df = pd.DataFrame([{'Study_Hours': study_hours, 'Attendance_Rate': attendance_rate}])
    
    if st.button("Predict Exam Score"):
        model = models['regression'][model_choice]
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Exam Score ({model_choice}): {prediction:.2f}")

elif task == "Classification":
    model_choice = st.sidebar.selectbox(
        "Select Classification Model",
        list(models['classification'].keys())
    )
    study_hours = st.number_input("Study Hours", min_value=0.0, max_value=20.0, value=5.0)
    attendance_rate = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=80.0)
    gender = st.selectbox("Gender", ("Male", "Female"))
    internet_access = st.selectbox("Internet Access", ("Yes", "No"))
    
    input_data = {
        'Study_Hours': study_hours,
        'Attendance_Rate': attendance_rate,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Internet_Access_Yes': 1 if internet_access == 'Yes' else 0
    }
    input_df = pd.DataFrame([input_data])
    
    # Ensure column order matches training data for models that are sensitive to it
    # For simplicity, assuming the order created by get_dummies was: Study_Hours, Attendance_Rate, Gender_Male, Internet_Access_Yes
    # In a real app, you might want to save the feature columns order.
    feature_cols = ['Study_Hours', 'Attendance_Rate', 'Gender_Male', 'Internet_Access_Yes']
    input_df = input_df[feature_cols]

    if st.button("Predict Pass/Fail"):
        model = models['classification'][model_choice]
        prediction = model.predict(input_df)[0]
        result = "Pass" if prediction == 1 else "Fail"
        st.success(f"Prediction ({model_choice}): {result}")

elif task == "Clustering":
    model_choice = st.sidebar.selectbox(
        "Select Clustering Model",
        list(models['clustering'].keys())
    )
    study_hours = st.number_input("Study Hours", min_value=0.0, max_value=20.0, value=5.0)
    attendance_rate = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=80.0)
    exam_score = st.number_input("Exam Score", min_value=0.0, max_value=100.0, value=75.0)
    
    input_df = pd.DataFrame([{'Study_Hours': study_hours, 'Attendance_Rate': attendance_rate, 'Exam_Score': exam_score}])
    
    if st.button("Assign Cluster"):
        model = models['clustering'][model_choice]
        # Scale the input data using the loaded scaler
        scaled_input = scaler.transform(input_df[['Study_Hours', 'Attendance_Rate', 'Exam_Score']])
        cluster_assignment = model.predict(scaled_input)[0]
        st.success(f"Assigned Cluster ({model_choice}): {cluster_assignment}")
