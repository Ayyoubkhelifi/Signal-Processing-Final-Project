import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import time

class WOA:
    def __init__(self, population_size=3, max_iter=5):  # Further reduced parameters
        self.population_size = population_size
        self.max_iter = max_iter
        self.best_position = None
        self.best_score = float('inf')
        
        # Reduced parameter ranges
        self.param_bounds = {
            'learning_rate': (0.001, 0.01),
            'batch_size': (64, 128),  # Increased minimum batch size
            'neurons': (16, 32)  # Reduced neuron range
        }

    def _create_model(self, params, input_shape):
        from models.model import create_cnn_model
        model = create_cnn_model(
            input_shape=input_shape,
            learning_rate=params[0],
            neurons=int(params[2])
        )
        return model

    def optimize(self, X_train, y_train, X_val, y_val, input_shape):
        """
        Optimize hyperparameters using WOA algorithm with early termination
        """
        print("Starting WOA optimization...")
        start_time = time.time()
        
        # Initialize population
        population = np.zeros((self.population_size, len(self.param_bounds)))
        for i in range(self.population_size):
            for j, (param, (low, high)) in enumerate(self.param_bounds.items()):
                population[i, j] = np.random.uniform(low, high)

        # Early stopping criteria
        no_improve_count = 0
        best_score_threshold = 0.01  # Stop if improvement is less than 1%
        
        for iteration in range(self.max_iter):
            iteration_start = time.time()
            print(f"\nOptimization iteration {iteration + 1}/{self.max_iter}")
            
            prev_best_score = self.best_score
            
            for i, individual in enumerate(population):
                print(f"Training model {i + 1}/{self.population_size}")
                
                model = self._create_model(individual, input_shape)
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=2,  # Further reduced patience
                    restore_best_weights=True
                )
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=10,  # Further reduced epochs
                    batch_size=int(individual[1]),
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                score = min(history.history['val_loss'])
                print(f"Val loss: {score:.4f}")
                
                if score < self.best_score:
                    improvement = (self.best_score - score) / self.best_score
                    self.best_score = score
                    self.best_position = individual.copy()
                    print(f"New best score: {self.best_score:.4f}")
                    
                    # Check for early termination
                    if improvement < best_score_threshold:
                        no_improve_count += 1
                    else:
                        no_improve_count = 0
                        
                    if no_improve_count >= 2:  # Stop if no significant improvement for 2 iterations
                        print("Early termination: No significant improvement")
                        return self._get_best_params()
            
            # Update population using simplified WOA
            a = 2 * (1 - iteration / self.max_iter)  # Linear decrease
            for i in range(self.population_size):
                if np.random.rand() < 0.5:
                    # Move towards best solution
                    population[i] = population[i] + np.random.rand() * (self.best_position - population[i])
                else:
                    # Random search
                    population[i] = population[i] + np.random.randn(len(self.param_bounds)) * 0.1
            
            # Clip values to parameter bounds
            for j, (param, (low, high)) in enumerate(self.param_bounds.items()):
                population[:, j] = np.clip(population[:, j], low, high)
            
            iteration_time = time.time() - iteration_start
            print(f"Iteration time: {iteration_time:.1f}s")

        return self._get_best_params()

    def _get_best_params(self):
        return {
            'learning_rate': self.best_position[0],
            'batch_size': int(self.best_position[1]),
            'neurons': int(self.best_position[2])
        }, self.best_score
