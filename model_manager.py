# model_manager.py
from flask import Flask, request, jsonify
import mysql.connector
import io
import base64
import logging
import time
import os

app = Flask(__name__)

# Database configuration
db_config = {
    'user': 'root',
    'password': 'yourpassword',
    'host': '127.0.0.1',
    'port': '3307',
    'database': 'model_db'
}

@app.route('/model/<int:model_id>', methods=['PUT'])
def update_model(model_id):
    logging.info('Updating Model started')
    try:
        model_details = request.get_json()
        updates = []

        # Construct the update statement based on provided details
        if 'name' in model_details:
            updates.append("name = %s")
        if 'dataset_name' in model_details:
            updates.append("dataset_name = %s")
        if 'created_at' in model_details:
            updates.append("created_at = %s")
        if 'accuracy' in model_details:
            updates.append("accuracy = %s")

        if not updates:
            return jsonify({'error': 'No valid fields to update'}), 400

        update_statement = "UPDATE models SET " + ", ".join(updates) + " WHERE id = %s"
        update_values = [model_details.get(field.split(" ")[0]) for field in updates]
        update_values.append(model_id)

        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute(update_statement, update_values)
        connection.commit()
        
        if cursor.rowcount == 0:
            return jsonify({'error': 'Model not found or no changes made'}), 404
        logging.info('Updating Model Finished')
        return jsonify({'message': 'Model updated successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()
        
@app.route('/model', methods=['GET'])
def get_model():
    try:
        model_name = request.args.get('name')
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        # Select the model weights blob from the database
        cursor.execute("SELECT weights FROM model_weights JOIN models ON models.id = model_weights.model_id WHERE models.name = %s", (model_name,))
        model_weights = cursor.fetchone()

        if model_weights:
            # model_weights[0] is a blob of weights
            # Convert the binary data to a base64 encoded string
            weights_base64_encoded = base64.b64encode(model_weights[0]).decode('utf-8')
            return jsonify({'name': model_name, 'data': weights_base64_encoded}), 200
        else:
            return jsonify({'error': 'Model not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200


@app.route('/model', methods=['DELETE'])
def delete_model():
    model_name = request.args.get('name')

    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    try:
        # Delete model weights first (if there's a foreign key constraint)
        cursor.execute("DELETE FROM model_weights WHERE model_id IN (SELECT id FROM models WHERE name = %s)", (model_name,))
        
        # Delete model metadata
        cursor.execute("DELETE FROM models WHERE name = %s", (model_name,))
        connection.commit()

        return jsonify({'message': 'Model deleted successfully'}), 200
    except Exception as e:
        connection.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        connection.close()


if __name__ == '__main__':
    # Get the port number from the environment variable PORT, default to 5001 if not set
    start_time = time.time()
    logging.info(f"Application started at {start_time}")
    port = int(os.environ.get("PORT", 8084))
    app.run(host='0.0.0.0', port=port)
