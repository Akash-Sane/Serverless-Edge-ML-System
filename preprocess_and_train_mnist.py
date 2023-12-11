# Flask application: predict_train_mnist.py

from flask import Flask, jsonify, request
import tensorflow as tf
from google.cloud import storage
import mysql.connector
import os
import logging
import uuid
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
gcs_bucket_name = 'akash-model-storage' # My Bucket Name


def create_cnn_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def upload_model_to_gcs(model, model_name):
    client = storage.Client()
    bucket = client.bucket(gcs_bucket_name)
    blob = bucket.blob(model_name)
    
    model.save(model_name)
    blob.upload_from_filename(model_name)

    # Clean up - remove the local file
    os.remove(model_name)
    
    return blob.public_url

def save_model_path_to_db(model_name, dataset_name, architecture, accuracy, loss, gcs_model_path):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    try:
        cursor.execute("""
            INSERT INTO models (name, dataset_name, architecture, accuracy, loss, model_path)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (model_name, dataset_name, architecture, accuracy, loss, gcs_model_path))
        connection.commit()
    finally:
        cursor.close()
        connection.close()


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200


@app.route('/train_mnist', methods=['GET'])
def train_mnist():
    logging.info('Training MNIST model started')
    try:
        (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255

        model = create_cnn_model((28, 28, 1), 10)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train the model
        history = model.fit(x_train, y_train, batch_size=32, epochs=15)

        best_accuracy = max(history.history['accuracy'])
        best_loss = min(history.history['loss'])

        # Generate a unique name for the model
        model_id = str(uuid.uuid4())
        gcs_model_name = f"mnist_model_{model_id}.h5"
        
        # Upload the model to GCS
        gcs_model_path = upload_model_to_gcs(model, gcs_model_name)
        
        # Save the GCS model path to the database
        save_model_path_to_db('MNIST_CNN', 'MNIST', 'CNN', best_accuracy, best_loss, gcs_model_path)
        logging.info('MNIST model training completed')
        return jsonify({'message': f'MNIST model trained and saved to GCS. Model path: {gcs_model_path}'})

    except Exception as e:
        logging.error("Error in training MNIST model: %s", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get the port number from the environment variable PORT, default to 5001 if not set
    start_time = time.time()
    logging.info(f"Application started at {start_time}")
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)