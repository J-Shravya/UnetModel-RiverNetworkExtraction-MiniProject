from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import io
import base64
from tensorflow.keras.models import load_model

# Initialize Flask application
app = Flask(__name__)

# Load your trained U-Net model
model = load_model('unetfinalized_model.h5')

# Define input size
input_width = 128
input_height = 128

# Define preprocessing function
def preprocess_image(image):
    resized_image = image.resize((input_width, input_height))
    image_array = np.array(resized_image)
    normalized_image = image_array / 255.0  # Adjust normalization based on your training data
    return normalized_image

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    preprocessed_image = preprocess_image(image)
    segmented_image = model.predict(np.expand_dims(preprocessed_image, axis=0))
    segmented_mask = (segmented_image > 0.5).astype(np.uint8)
    segmented_mask_pil = Image.fromarray(segmented_mask[0, :, :, 0] * 255)
    
    # Convert image to base64 string
    buffered = io.BytesIO()
    segmented_mask_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return jsonify({'result': img_str})

if __name__ == '__main__':
    app.run(debug=True)
