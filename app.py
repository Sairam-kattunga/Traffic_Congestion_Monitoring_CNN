import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- 1. Load the trained model ---
MODEL_PATH = "models/resnet50_transfer_model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- 2. Define the class names ---
CLASS_NAMES = ["High_Congestion", "Low_Congestion"]

# --- 3. Create the prediction function (no changes here) ---
def predict(image):
    if model is None:
        return {"Error": "Model could not be loaded. Please check file path."}
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array_batch = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input(image_array_batch.astype(np.float32))
    predictions = model.predict(preprocessed_image)
    confidence_scores = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    return confidence_scores

# --- 4. Build the Gradio Interface with gr.Blocks ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚦 Traffic Congestion Detector")
    gr.Markdown("Upload an image of a road to classify its traffic congestion level. This demo uses a ResNet50 model trained via transfer learning.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Traffic Image")
        output_label = gr.Label(num_top_classes=2, label="Prediction")

    submit_button = gr.Button("Classify")

    submit_button.click(
        fn=predict,
        inputs=image_input,
        outputs=output_label
    )

    gr.Markdown("<p style='text-align: center;'>A project to demonstrate computer vision skills using TensorFlow, Keras, and Gradio.</p>")

# --- 5. Launch the App (no changes here) ---
demo.launch()