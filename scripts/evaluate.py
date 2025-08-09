import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from preprocess_data import get_data_generators

# --- File Paths ---
MODELS_DIR = '../models/'
RESULTS_DIR = '../results/'

def plot_confusion_matrix(cm, class_names, model_name):
    """
    Plots a confusion matrix and saves it to a file.
    The matrix shows where the model got confused.
    """
    plt.figure(figsize=(8, 6))
    # Use seaborn to create a "heatmap"
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    # Make sure the 'results' directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Save the plot
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_confusion_matrix.png'))
    print(f"✅ Confusion matrix plot saved to {RESULTS_DIR}")
    plt.show()

def main():
    print("--- Starting Model Evaluation ---")
    
    # We only need the validation generator for evaluation
    _, validation_generator = get_data_generators()
    
    # Get the names of the classes (e.g., 'High_Congestion', 'Low_Congestion')
    class_names = list(validation_generator.class_indices.keys())
    
    # Get the true labels for all validation images
    y_true = validation_generator.classes
    
    # Find all trained model files in the 'models' directory
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.keras')]
    
    if not model_files:
        print("❌ No trained models found in the 'models' folder.")
        return
        
    for model_file in model_files:
        model_path = os.path.join(MODELS_DIR, model_file)
        # Get a clean name for the model from its filename
        model_name = model_file.replace('.keras', '').replace('_model', '')
        
        print(f"\n--- Evaluating: {model_name.upper()} ---")
        
        # Load the saved model
        model = tf.keras.models.load_model(model_path)
        
        # Get the model's predictions on the validation data
        y_pred_probs = model.predict(validation_generator)
        y_pred = np.argmax(y_pred_probs, axis=1) # Get the class with the highest probability
        
        # --- Print the Classification Report ---
        # This shows precision, recall, and f1-score
        print("\nClassification Report:")
        report = classification_report(y_true, y_pred, target_names=class_names)
        print(report)
        
        # --- Create and Plot the Confusion Matrix ---
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, class_names, model_name)
        
    print("\n--- Evaluation Finished ---")

if __name__ == '__main__':
    main()