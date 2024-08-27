from utils.tf2mlir_utils import *
import tensorflow as tf




# Define a simple model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(4,)),  # Example input shape (4,)
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3)  # Output layer with 3 units (for example)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

# Create and save the model
input_shape = (None, 4)  # Batch size can be variable, so use None
save_path = 'tmp_saved_model'

save_model(create_model, input_shape, save_path)
print(f'Model saved to {save_path}')
