from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import pytest
import json

app = Flask(__name__)

def unit_normalize(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

def train_and_save_models(data, labels):
    svm_model = SVC(kernel='linear')
    svm_model.fit(data, labels)
    joblib.dump(svm_model, 'm22aie232_svm_model.joblib')
    tree_model = DecisionTreeClassifier()
    tree_model.fit(data, labels)
    joblib.dump(tree_model, 'm22aie232_tree_model.joblib')
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    for solver in solvers:
        lr_model = LogisticRegression(solver=solver)
        lr_model.fit(data, labels)
        joblib.dump(lr_model, f'm22aie232_lr_model_{solver}.joblib')
def load_model(model_type):
    filename_mapping = {
        'svm': 'm22aie232_svm_model.joblib',
        'tree': 'm22aie232_tree_model.joblib',
        'lr': f'm22aie232_lr_model_{model_type}.joblib'
    }

    model_filename = filename_mapping.get(model_type)
    if model_filename is not None:
        return joblib.load(model_filename)
    else:
        return None
@app.route('/train', methods=['POST'])
def train_models():
    try:
        data = request.get_json()['data']
        labels = request.get_json()['labels']
        normalized_data = unit_normalize(data)
        train_and_save_models(normalized_data, labels)
        return jsonify({'message': 'Models trained and saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type):
    model = load_model(model_type)
    if model is None:
        return jsonify({'error': 'Invalid model type'}), 400
    try:
        data = request.get_json()['data']
        normalized_data = unit_normalize(data)
        prediction = model.predict(normalized_data)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
def test_loaded_model_is_lr():
    model = load_model('lr')
    assert isinstance(model, LogisticRegression)
def test_solver_name_match():
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    for solver in solvers:
        model = load_model(f'lr_model_{solver}')
        loaded_solver_name = model.get_params()['solver']
        assert loaded_solver_name == solver
if __name__ == '__main__':
    app.run(debug=True)