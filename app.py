from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import keras
import cv2
from PIL import Image
import io
app = Flask(__name__)

model = keras.models.load_model('new_model.h5')


# Define a function to preprocess input image

def preprocess_image_cv2(filepath):
    # Read the image
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 28x28 pixels
    img = cv2.resize(img, (28, 28))

    # Invert the image colors
    img = 255 - img

    # Normalize the pixel values
    img = img.astype('float32') / 255.0

    # Reshape the image to match the input shape of the model
    img = img.reshape(1, 28, 28, 1)

    return img


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_float_version():
    if request.method == 'POST':
        # Read the image file
        img = cv2.imread('captured_image.png', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = np.array(img).reshape(1, 28, 28, 1)
        img = img.astype('uint8') / 255

        prediction = model.predict(img)
        print(prediction)
        prediction = np.argmax(prediction)

        return jsonify({'prediction': str(prediction)})


@app.route('/capture', methods=['POST'])
def capture():
    if request.method == 'POST':
        data = request.json
        image_data = data['image']

        # Convert base64 image to a PIL Image object
        image = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))

        # Save the image as a PNG file
        image.save('captured_image.png')

        return 'Image captured and saved successfully'


if __name__ == '__main__':
    app.run(debug=True)
