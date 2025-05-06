from flask import Flask, render_template, request
from models.model import create_cnn_model  # or create_lstm_model
from preprocessing.preprocess import load_data, clean_data, normalize_data
from optimization.woa_optimizer import WOA
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    # Load and preprocess the data
    data = load_data('data/room_3')
    cleaned_data = clean_data(data)
    normalized_data = normalize_data(cleaned_data)
    
    # Define the model and optimize hyperparameters
    model = create_cnn_model(input_shape=(normalized_data.shape[1], 1))  # Adjust input shape
    
    # Example: Call WOA for optimization
    woa = WOA(population_size=10, max_iter=50)
    best_params, best_score = woa.optimize(fitness_function=None)  # Pass an actual fitness function here
    
    return f"Model trained successfully! Best Params: {best_params}, Score: {best_score}"

if __name__ == "__main__":
    app.run(debug=True)
