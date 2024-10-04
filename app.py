from flask import Flask, request, jsonify
import joblib
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Define the class names used for prediction
class_names = ['Bacterial dermatosis', 'Fungal infections', 'Healthy', 'Hypersensitivity allergic dermatosis', 'Mange']

# Define paths to your models
model_paths = {
    'Logistic Regression': 'logistic_regression_model.pkl',
    'Random Forest': 'random_forest_model.pkl',
    # 'SVM': 'svm_model.pkl'
}

# Load the deep learning model (update this to your model path)
dl_model = load_model('DL.keras')

def preprocess_dl_image(image, img_height=224, img_width=224):
    """Preprocess image for deep learning model."""
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)  # Read image from buffer
    img = cv2.resize(img, (img_height, img_width))  # Resize image
    img_array = img.astype('float32') / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def load_and_predict(image, model_paths, class_names, img_height=224, img_width=224):
    """Perform predictions using deep learning and machine learning models."""
    # Preprocess the image for the deep learning model
    dl_img_array = preprocess_dl_image(image, img_height, img_width)

    # Make prediction using the deep learning model
    dl_predictions = dl_model.predict(dl_img_array)
    dl_predicted_class_index = np.argmax(dl_predictions[0])
    dl_confidence = dl_predictions[0][dl_predicted_class_index]
    dl_predicted_class = class_names[dl_predicted_class_index]

    predictions = {
        'Deep Learning Model': {
            'predicted_class': dl_predicted_class,
            'confidence': float(dl_confidence)
        }
    }

    # Rewind the image file pointer for ML models
    image.seek(0)

    # Read and process image for machine learning models
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_height, img_width))  # Resize image
    img_flattened = img.flatten().reshape(1, -1)  # Flatten and reshape

    # Store predictions for ML models
    ml_predictions = []

    # Iterate over each model path, load the model, and make predictions
    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            continue  # Skip if model file doesn't exist
        model = joblib.load(model_path)  # Load the model
        pred = model.predict(img_flattened)  # Predict the class
        confidence = model.predict_proba(img_flattened).max()  # Get confidence score
        predicted_class = class_names[pred[0]]

        predictions[model_name] = {
            'predicted_class': predicted_class,
            'confidence': float(confidence)
        }

        # Store ML model predictions
        ml_predictions.append((predicted_class, confidence))

    # Aggregate predictions
    all_predictions = {dl_predicted_class: dl_confidence}  # Start with DL model prediction
    for predicted_class, confidence in ml_predictions:
        if predicted_class in all_predictions:
            all_predictions[predicted_class] += confidence  # Aggregate confidence
        else:
            all_predictions[predicted_class] = confidence

    # Find the model with the highest confidence
    highest_model_confidence = max(
        predictions.items(),
        key=lambda item: item[1]['confidence']
    )

    return {
        'highest_confidence_class': highest_model_confidence[1]['predicted_class'],
        'highest_confidence_value': highest_model_confidence[1]['confidence'],
        'prediction_count': sum(1 for _ in ml_predictions if _[0] == highest_model_confidence[1]['predicted_class']),
        'individual_predictions': predictions
    }

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to handle predictions."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Make predictions using the loaded models
    results = load_and_predict(file, model_paths, class_names)

    # Return the results as JSON
    return jsonify({
        'highest_confidence_class': results['highest_confidence_class'],
        'highest_confidence_value': results['highest_confidence_value'],
        'prediction_count': results['prediction_count'],
        'individual_predictions': results['individual_predictions']
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
