import tensorflow as tf
import os

# --- Constants you can change ---
IMG_WIDTH, IMG_HEIGHT = 224, 224 # The size of the images
BATCH_SIZE = 32 # How many images to process at a time

# --- File Paths ---
# This points to the dataset folder from the scripts folder's perspective
DATA_DIR = '../dataset/'

def get_data_generators():
    """
    Creates and returns train and validation data generators.
    """
    print("Reading images from:", os.path.abspath(DATA_DIR))

    # This tool from Keras will read images and apply transformations
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,           # Rescale pixel values from 0-255 to 0-1
        validation_split=0.2,     # Set aside 20% of images for testing
        rotation_range=20,        # Randomly rotate images
        width_shift_range=0.2,    # Randomly shift images horizontally
        height_shift_range=0.2,   # Randomly shift images vertically
        shear_range=0.2,          # Apply shearing transformations
        zoom_range=0.2,           # Randomly zoom into images
        horizontal_flip=True,     # Randomly flip images horizontally
        fill_mode='nearest'
    )

    # --- Training Data Generator ---
    # Reads from the DATA_DIR and uses the 'training' subset (80%)
    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical', # For multi-class classification
        subset='training',
        shuffle=True
    )

    # --- Validation Data Generator ---
    # Reads from the DATA_DIR and uses the 'validation' subset (20%)
    # Note: It only rescales validation images, no other augmentation is applied.
    validation_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print("✅ Data generators created successfully!")
    print(f"Found {train_generator.samples} training images belonging to {train_generator.num_classes} classes.")
    print(f"Found {validation_generator.samples} validation images.")
    
    return train_generator, validation_generator

# This part allows us to test the script directly
if __name__ == '__main__':
    print("--- Testing the data preprocessing script ---")
    get_data_generators()