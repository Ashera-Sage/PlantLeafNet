from PIL import Image
import io
import tensorflow as tf
import numpy as np
import json
from flask import Flask, render_template, request

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(
    model_path="efficientnet_plant_disease_quantized.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No image uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No image selected"

    # Read and preprocess image
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = image.resize((256, 256))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # Model prediction
    interpreter.set_tensor(
        input_details[0]["index"],
        image_array
    )

    interpreter.invoke()

    predictions = interpreter.get_tensor(
        output_details[0]["index"]
    )

    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100

    disease_name = class_names[predicted_index]

    # Confidence threshold
    if confidence < 70:
        result = "Uncertain Prediction"
        explanation = "Please upload a clearer image of the leaf."
    else:
        result = disease_name
        explanation = "Prediction completed successfully."

    return render_template(
        "index.html",
        result=result,
        confidence=round(confidence, 2),
        explanation=explanation
    )

if __name__ == "__main__":
    app.run(debug=True)