from flask import Flask, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import mysql.connector
import os
import uuid
from google.cloud import storage
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
gcs_bucket_name = 'akash-model-storage'  # My GCS bucket name

def create_model(input_shape, num_classes):
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
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


@app.route('/train_imagenet', methods=['GET'])
def train_imagenet():
    logging.info('Training MNIST model started')
    try:
        # Path to the ImageNet-200 data
        train_data_dir = 'C:/NCI/RIC/Project/tiny-imagenet-200/tiny-imagenet-200/train/'
        validation_data_dir = 'C:/NCI/RIC/Project/tiny-imagenet-200/tiny-imagenet-200/val/'
        num_classes = 200
        input_shape = (64, 64, 3)

        # Data generators
        datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)

        train_generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(input_shape[0], input_shape[1]),
            batch_size=64,
            class_mode='categorical')

        validation_generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(input_shape[0], input_shape[1]),
            batch_size=64,
            class_mode='categorical')

        # Create and compile the model
        model = create_model(input_shape, num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(
            train_generator,
            epochs=15,  # Adjust epochs as needed
            validation_data=validation_generator)

        # Generate a unique name for the model
        model_id = str(uuid.uuid4())
        gcs_model_name = f"imagenet_model_{model_id}.h5"
        
        # Upload the model to GCS
        gcs_model_path = upload_model_to_gcs(model, gcs_model_name)
        
        # Save the GCS model path to the database
        save_model_path_to_db('ImageNet-200_CNN', 'ImageNet-200', 'ResNet_CNN', max(history.history['val_accuracy']), min(history.history['val_loss']), gcs_model_path)
        logging.info('ImageNet-200 model training completed')
        return jsonify({'message': f'ImageNet-200 model trained and saved to GCS. Model path: {gcs_model_path}'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Get the port number from the environment variable PORT, default to 8082 if not set
    start_time = time.time()
    logging.info(f"Application started at {start_time}")
    port = int(os.environ.get("PORT", 8082))
    app.run(host='0.0.0.0', port=port)
