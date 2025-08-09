import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from preprocess_data import get_data_generators # We import our own data loader
import os

# --- Constants ---
EPOCHS = 15 # How many times the model sees the entire dataset
MODELS_DIR = '../models/' # Go up one level to find the 'models' folder
MODEL_NAME = 'custom_cnn_model.keras' # The name for our saved model

def build_custom_cnn(num_classes):
    """Defines and builds a simple CNN model from scratch."""
    
    # A Sequential model means layers are stacked one after another
    model = Sequential([
        Input(shape=(224, 224, 3)), # Defines the image size
        
        # 1st Convolutional Block: Finds basic edges and patterns
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # 2nd Convolutional Block: Finds more complex patterns
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # 3rd Convolutional Block: Finds even more complex features
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Prepares the data for the final classification layers
        Flatten(),
        
        # The 'brain' part that makes the final decision
        Dense(128, activation='relu'),
        Dropout(0.5), # A technique to prevent the model from just memorizing
        Dense(num_classes, activation='softmax') # Output layer
    ])
    
    # Compile the model, preparing it for training
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("✅ Custom CNN model built successfully.")
    model.summary() # Print a summary of the model architecture
    return model

def main():
    # Make sure the 'models' directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Get the data generators from our other script
    train_generator, validation_generator = get_data_generators()
    
    # Get the number of classes (e.g., 2 for High/Low) automatically
    num_classes = train_generator.num_classes
    
    # Build the model
    model = build_custom_cnn(num_classes)
    
    print("\n--- Starting Training for Custom CNN ---")
    
    # The main training command
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )
    
    # Save the final trained model
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    model.save(model_path)
    print(f"\n✅ Model saved successfully to {model_path}")
    print("--- Training Finished ---")

# This ensures the main() function runs when you execute the script
if __name__ == '__main__':
    main()