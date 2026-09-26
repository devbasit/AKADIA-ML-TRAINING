from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models and scaler
# In a real application, consider using a global variable or app context for efficiency
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

@app.route('/predict/regression', methods=['POST'])
def predict_regression_flask():
    data = request.get_json()
    study_hours = data.get('study_hours')
    attendance_rate = data.get('attendance_rate')
    model_name = data.get('model_name')

    if not all([study_hours, attendance_rate, model_name]):
        return jsonify({"error": "Missing data points or model name"}), 400
    
    model = regression_models.get(model_name)
    if not model:
        return jsonify({"error": "Invalid regression model name"}), 400
    
    df = pd.DataFrame([{
        'Study_Hours': study_hours,
        'Attendance_Rate': attendance_rate
    }])
    
    prediction = model.predict(df)[0]
    return jsonify({"task": "regression", "model": model_name, "predicted_exam_score": round(prediction, 2)})

@app.route('/predict/classification', methods=['POST'])
def predict_classification_flask():
    data = request.get_json()
    study_hours = data.get('study_hours')
    attendance_rate = data.get('attendance_rate')
    gender = data.get('gender')
    internet_access = data.get('internet_access')
    model_name = data.get('model_name')

    if not all([study_hours, attendance_rate, gender, internet_access, model_name]):
        return jsonify({"error": "Missing data points or model name"}), 400

    model = classification_models.get(model_name)
    if not model:
        return jsonify({"error": "Invalid classification model name"}), 400
    
    data_dict = {
        'Study_Hours': [study_hours],
        'Attendance_Rate': [attendance_rate],
        'Gender': [gender],
        'Internet_Access': [internet_access]
    }
    df = pd.DataFrame(data_dict)

    df['Gender_Male'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['Internet_Access_Yes'] = df['Internet_Access'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    feature_cols = ['Study_Hours', 'Attendance_Rate', 'Gender_Male', 'Internet_Access_Yes']
    input_features = df[feature_cols]

    prediction = model.predict(input_features)[0]
    result = "Pass" if prediction == 1 else "Fail"
    return jsonify({"task": "classification", "model": model_name, "predicted_pass_fail": result})

@app.route('/predict/clustering', methods=['POST'])
def predict_clustering_flask():
    data = request.get_json()
    study_hours = data.get('study_hours')
    attendance_rate = data.get('attendance_rate')
    exam_score = data.get('exam_score')
    model_name = data.get('model_name')

    if not all([study_hours, attendance_rate, exam_score, model_name]):
        return jsonify({"error": "Missing data points or model name"}), 400

    model = clustering_models.get(model_name)
    if not model:
        return jsonify({"error": "Invalid clustering model name"}), 400
    
    df = pd.DataFrame([{
        'Study_Hours': study_hours,
        'Attendance_Rate': attendance_rate,
        'Exam_Score': exam_score
    }])

    scaled_input = scaler.transform(df[['Study_Hours', 'Attendance_Rate', 'Exam_Score']])
    
    cluster_assignment = model.predict(scaled_input)[0]
    return jsonify({"task": "clustering", "model": model_name, "assigned_cluster": int(cluster_assignment)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
