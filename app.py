from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Global variables for model and features
model = None
feature_names = None

def load_model():
    """Load the model and features safely with error handling"""
    global model, feature_names
    
    try:
        model_path = 'random_forest_tire_lifespan_model.pkl'
        feature_path = 'features.pkl'
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Features file not found: {feature_path}")
        
        # Load the model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        # Load the features
        with open(feature_path, 'rb') as file:
            feature_names = pickle.load(file)
        
        # Verify model has predict method
        if not hasattr(model, 'predict'):
            raise AttributeError("Loaded model doesn't have predict method")
        
        print("Model and features loaded successfully!")
        print(f"Features: {feature_names}")
        return True
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.route('/')
def home():
    """Render the home page"""
    if feature_names is None:
        return "Error: Model not properly loaded", 500
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Get values from the form
        features = []
        for feature in feature_names:
            value = request.form.get(feature)
            if value is None:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            try:
                features.append(float(value))
            except ValueError:
                return jsonify({'error': f'Invalid value for feature: {feature}'}), 400

        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]

        return jsonify({
            'success': True,
            'prediction': round(float(prediction), 2),
            'message': f'Predicted tire lifespan: {round(float(prediction), 2)} units'
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not load_model():
        print("Failed to load model. Exiting.")
        exit(1)
    app.run(debug=True)