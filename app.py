import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# --- 1. Load the trained model ---
# Make sure the model file is in the 'models' directory
MODEL_PATH = "models/resnet50_transfer_model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a dummy function if model loading fails, so app still runs
    def model(x): return None 

# --- 2. Define the class names ---
CLASS_NAMES = ["High_Congestion", "Low_Congestion"]

# --- 3. Create the prediction function ---
def predict(image):
    """
    Takes a PIL image, preprocesses it, and returns a dictionary of predictions.
    """
    if model is None:
        return {"Error": "Model could not be loaded. Please check the file path."}

    # Preprocess the image to match the model's input requirements
    # a. Resize to 224x224
    image = image.resize((224, 224))
    # b. Convert to numpy array and rescale pixel values
    image_array = np.array(image) / 255.0
    # c. Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    # Make a prediction
    predictions = model.predict(image_array)
    
    # Format the output as a dictionary {<label>: <confidence>}
    confidence_scores = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    
    return confidence_scores

# --- 4. Build the Gradio Interface ---
# This creates the UI components
demo = gr.Interface(
    fn=predict,                                     # The function to call on input
    inputs=gr.Image(type="pil"),                    # Input type is an image
    outputs=gr.Label(num_top_classes=2),            # Output is a label with top 2 classes
    title="🚦 Traffic Congestion Detector",
    description="Upload an image of a road to classify its traffic congestion level. This demo uses a ResNet50 model trained via transfer learning.",
    examples=[
        # Add paths to some example images from your dataset here if you like
        "dataset/High_Congestion/1.jpg",
        "dataset/Low_Congestion/1.jpg"
    ],
    article="<p style='text-align: center;'>Built with TensorFlow and Gradio. This is a portfolio project to demonstrate deep learning skills in computer vision.</p>"
)

# --- 5. Launch the App ---
if __name__ == "__main__":
    demo.launch() # Launch the web service