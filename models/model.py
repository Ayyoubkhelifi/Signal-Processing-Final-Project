from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam

def create_cnn_model(input_shape, learning_rate=0.001, neurons=32):
    # Smaller, more efficient architecture
    inputs = Input(shape=input_shape)
    x = Conv1D(neurons, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(4)(x)  # Increased pool size for faster reduction
    x = Dropout(0.2)(x)
    x = Conv1D(neurons//2, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(4)(x)
    x = Flatten()(x)
    x = Dense(neurons//2, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'],
                 run_eagerly=False)  # Disable eager execution for speed
    return model
