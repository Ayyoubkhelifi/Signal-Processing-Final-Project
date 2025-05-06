from models.model import create_cnn_model  # or create_lstm_model
from preprocessing.preprocess import load_data, clean_data, normalize_data
from optimization.woa_optimizer import WOA

def main():
    # Load and preprocess data
    data = load_data('data/room_3')
    cleaned_data = clean_data(data)
    normalized_data = normalize_data(cleaned_data)
    
    # Define your model
    model = create_cnn_model(input_shape=(normalized_data.shape[1], 1))  # Example CNN model
    
    # Fit model (simplified)
    # model.fit(normalized_data, labels)  # Add labels accordingly
    
    # Optimize hyperparameters with WOA
    woa = WOA(population_size=10, max_iter=50)
    best_params, best_score = woa.optimize(fitness_function=None)  # Define fitness function
    
    print(f"Best Params: {best_params}, Best Score: {best_score}")

if __name__ == "__main__":
    main()
