import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained model
model = load_model('model.keras')

# Function to preprocess the input image
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    image_resized = cv2.resize(image, (128, 128))
    # Normalize pixel values
    image_rescaled = image_resized / 255.0
    # Return the preprocessed image
    return image_rescaled

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read the image file
        image_stream = file.read()
        nparr = np.frombuffer(image_stream, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Reshape the image to match the input shape expected by the model
        input_image = np.expand_dims(preprocessed_image, axis=0)

        # Make predictions using the model
        prediction = model.predict(input_image)

        # The prediction will be a probability, convert it to a binary label
        if prediction[0][0] >= 0.5:
            result = "Real"
        else:
            result = "Fake"
            
            # Save the uploaded image
        file_path = os.path.join('uploads', file.filename)
        cv2.imwrite(file_path, image)

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
