from flask import Flask, render_template, request, jsonify, Response
from models.model import create_cnn_model
from preprocessing.preprocess import load_data, clean_data, normalize_data
from optimization.woa_optimizer import WOA
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json
import time

app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'))

def load_labels(room):
    labels = []
    room_dir = os.path.join(os.path.dirname(__file__), '..', 'data', room)
    
    for subfolder in os.listdir(room_dir):
        subfolder_path = os.path.join(room_dir, subfolder)
        if os.path.isdir(subfolder_path):
            label_file = os.path.join(subfolder_path, 'label.csv')
            if os.path.exists(label_file):
                try:
                    print(f"Loading labels from {label_file}")
                    label_data = pd.read_csv(label_file)
                    labels.append(label_data)
                except Exception as e:
                    print(f"Error reading {label_file}: {e}")
    
    if not labels:
        raise ValueError("No label files were loaded")
        
    return pd.concat(labels, ignore_index=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Get form data
        room = request.form.get('room', 'room_1')
        model_type = request.form.get('model_type', 'cnn')
        
        def generate_progress():
            # Step 1: Data Loading
            yield json.dumps({
                'step': 1,
                'status': 'Loading data...',
                'progress': 25
            }) + '\n'
            
            # Load and preprocess the data
            print(f"Loading data from {room}")
            data = load_data(f'data/{room}', n_components=100)
            print(f"Data shape after PCA: {data.shape}")
            
            # Step 2: Preprocessing
            yield json.dumps({
                'step': 2,
                'status': 'Preprocessing data...',
                'progress': 50
            }) + '\n'
            
            # Load labels
            labels = load_labels(room)
            print(f"Labels shape: {labels.shape}")
            
            # Make sure data and labels have matching lengths
            min_len = min(len(data), len(labels))
            X = data[:min_len]
            y = labels.iloc[:min_len, 0].values
            
            # Split data into train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Reshape data for CNN input
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            
            # Step 3: Model Training
            yield json.dumps({
                'step': 3,
                'status': 'Training model...',
                'progress': 75
            }) + '\n'
            
            # Define input shape for the model
            input_shape = (X_train.shape[1], 1)
            print(f"Input shape: {input_shape}")
            
            # Initialize and run WOA optimization
            woa = WOA(population_size=3, max_iter=5)
            best_params, best_score = woa.optimize(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                input_shape=input_shape
            )
            
            # Step 4: Final optimization
            yield json.dumps({
                'step': 4,
                'status': 'Optimization complete',
                'progress': 100,
                'result': {
                    'status': 'success',
                    'details': {
                        'best_params': best_params,
                        'best_score': float(best_score) if best_score is not None else None,
                        'data_shape': X.shape,
                        'explained_variance': best_score
                    }
                }
            }) + '\n'

        return Response(generate_progress(), mimetype='text/event-stream')
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
