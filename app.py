import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# IMPORTANT: Import the specific preprocessing function for the ResNet50 model
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- 1. Load the trained model ---
# Make sure the model file is in the 'models' directory
MODEL_PATH = "models/resnet50_transfer_model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- 2. Define the class names ---
CLASS_NAMES = ["High_Congestion", "Low_Congestion"]

# --- 3. Create the UPDATED prediction function ---
def predict(image):
    """
    Takes a PIL image, preprocesses it correctly for ResNet50,
    and returns a dictionary of predictions.
    """
    if model is None:
        return {"Error": "Model could not be loaded. Please check the file path."}

    # Preprocess the image
    # a. Resize to the target size
    image = image.resize((224, 224))

    # b. Convert the image to a NumPy array
    image_array = np.array(image)
    
    # c. Add a batch dimension for the model
    image_array_batch = np.expand_dims(image_array, axis=0)

    # d. **CRITICAL STEP**: Use the official ResNet50 preprocessing function.
    # This correctly formats the colors and pixel values.
    preprocessed_image = preprocess_input(image_array_batch.astype(np.float32))

    # Make a prediction
    predictions = model.predict(preprocessed_image)
    
    # Format the output as a dictionary {<label>: <confidence>}
    confidence_scores = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    
    return confidence_scores

# --- 4. Build the Gradio Interface (No changes here) ---
# AFTER
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Traffic Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="🚦 Traffic Congestion Detector",
    description="Upload an image of a road to classify its traffic congestion level. This demo uses a ResNet50 model trained via transfer learning.",
    article="<p style='text-align: center;'>A project to demonstrate computer vision skills using TensorFlow, Keras, and Gradio.</p>"
)

# --- 5. Launch the App ---
if __name__ == "__main__":
    demo.launch()