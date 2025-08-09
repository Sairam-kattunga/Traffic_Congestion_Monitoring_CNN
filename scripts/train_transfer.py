import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from preprocess_data import get_data_generators # Import our data loader
import os
import argparse # To allow choosing a model from the command line

# --- Constants ---
EPOCHS = 15
MODELS_DIR = '../models/'
IMG_WIDTH, IMG_HEIGHT = 224, 224

def build_transfer_model(model_name, num_classes):
    """
    Builds a model using a pre-trained base and adds a custom head.
    """
    input_tensor = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    
    # Choose the pre-trained model base
    if model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    elif model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
    else: # Default to MobileNetV2
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)

    # Freeze the layers of the pre-trained model so they don't get re-trained
    base_model.trainable = False
    
    # Add our new custom layers on top of the pre-trained base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    # The final prediction layer
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # This is the final model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(f"✅ {model_name.upper()} model built for transfer learning.")
    model.summary()
    return model

def main(model_choice):
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    train_generator, validation_generator = get_data_generators()
    num_classes = train_generator.num_classes
    
    model = build_transfer_model(model_choice, num_classes)
    
    print(f"\n--- Starting Transfer Learning for {model_choice.upper()} ---")
    
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )
    
    model_path = os.path.join(MODELS_DIR, f'{model_choice}_transfer_model.keras')
    model.save(model_path)
    print(f"\n✅ Model saved successfully to {model_path}")
    print("--- Training Finished ---")

if __name__ == '__main__':
    # This part allows us to choose which model to train from the command line
    # Example: python train_transfer.py --model vgg16
    parser = argparse.ArgumentParser(description='Train a model using transfer learning.')
    parser.add_argument('--model', type=str, default='vgg16', 
                        choices=['vgg16', 'resnet50', 'mobilenet'],
                        help='The base model to use for transfer learning.')
    args = parser.parse_args()
    main(args.model)