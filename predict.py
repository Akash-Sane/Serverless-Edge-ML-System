from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from google.cloud import storage
import mysql.connector
from PIL import Image
import io
import os
import tempfile
import logging
import time

app = Flask(__name__)

# Database configuration
db_config = {
    'user': 'root',
    'password': 'yourpassword',
    'host': '127.0.0.1',
    'port': '3307',
    'database': 'model_db' 
}

def download_model_from_gcs(bucket_name, model_object_name):
    """Downloads a model from Google Cloud Storage and returns a BytesIO object."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_object_name)

    model_stream = io.BytesIO()
    blob.download_to_file(model_stream)
    model_stream.seek(0)
    return model_stream

def load_model(model_id):
    # Database connection
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    # Fetch the model path and type from the database
    cursor.execute("SELECT model_path, dataset_name FROM models WHERE id = %s", (model_id,))
    model_record = cursor.fetchone()
    cursor.close()
    connection.close()

    if not model_record:
        raise ValueError(f"No model found for model_id {model_id}")

    model_url, model_type = model_record  # Unpack model URL and the model type

    # Extract the object name from the URL
    object_name = model_url.split('/')[-1]

    # Download the model
    model_stream = download_model_from_gcs('akash-model-storage', object_name)

    # Write the model stream to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        temp_file.write(model_stream.read())
        temp_file_path = temp_file.name

    # Load and return the model and its type
    model = tf.keras.models.load_model(temp_file_path)
    logging.info('Prediction Complete')
    return model, model_type

def preprocess_image(image, dataset_name):
    if dataset_name == 'MNIST':
        image = image.convert('L')  # Convert to grayscale
        target_size = (28, 28)
    elif dataset_name == 'CIFAR-10':
        target_size = (32, 32)
    elif dataset_name == 'ImageNet-200':
        target_size = (64, 64)  # Adjust as needed for your specific ImageNet model
    else:
        raise ValueError("Unknown model type")

    # Resize and preprocess the image
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image = np.asarray(image)
    image = image / 255.0

    if dataset_name == 'MNIST':
        image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale

    return np.expand_dims(image, axis=0)  # Add batch dimension

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    logging.info('Prediction started')
    try:
        model_id = request.args.get('model_id', type=int)
        if not model_id:
            return jsonify({'error': 'No model_id provided.'}), 400

        model, model_type = load_model(model_id) 
    except Exception as e:
        return jsonify({'error': str(e)}), 500        

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided.'}), 400
        image_file = request.files['image']
        if image_file:
            image = Image.open(image_file).convert('RGB')
            processed_image = preprocess_image(image, model_type) 
            predictions = model.predict(processed_image)
            return jsonify({'predictions': predictions.tolist()})
        else:
            return jsonify({'error': 'Image file is empty.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get the port number from the environment variable PORT, default to 5001 if not set
    start_time = time.time()
    logging.info(f"Application started at {start_time}")
    port = int(os.environ.get("PORT", 8083))
    app.run(host='0.0.0.0', port=port)