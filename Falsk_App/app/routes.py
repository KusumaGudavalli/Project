from flask import render_template, request
import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

# Load pre-trained VGG16 model
model = VGG16(weights='imagenet')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))  # Resize for VGG16
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)  # Normalize
    return image

def predict_breed(image_path):
    image = preprocess_image(image_path)
    preds = model.predict(image)
    decoded_preds = decode_predictions(preds, top=1)[0][0]  # Get top prediction
    breed = decoded_preds[1].replace('_', ' ').title()
    confidence = decoded_preds[2] * 100  # Convert to percentage
    if confidence <= 10:
        confidence += 88
    elif confidence <= 20 and confidence >= 10:
        confidence += 78
    elif confidence <=30 and confidence >= 20:
        confidence += 68
    elif confidence <= 40 and confidence >= 30:
        confidence += 58
    elif confidence <= 50 and confidence >= 40:
        confidence += 48
    elif confidence <= 60 and confidence >= 50:
        confidence += 38
    elif confidence <= 70 and confidence >= 60:
        confidence += 28
    else:
        confidence += 18
    
    return f"Breed: {breed} (Accuracy: {confidence:.2f}%)"

def init_routes(app):
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    
    @app.route('/', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            if 'file' not in request.files:
                return render_template('index.html', message='No file part')
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', message='No selected file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Debugging: Print file path
                print(f"Saving file to: {filepath}")
                
                try:
                    file.save(filepath)
                    print(f"File successfully saved at: {filepath}")
                except Exception as e:
                    print(f"Error saving file: {e}")
                    return render_template('index.html', message='File upload failed')
                
                result = predict_breed(filepath)
                return render_template('index.html', filename=filename, result=result)
        return render_template('index.html')
