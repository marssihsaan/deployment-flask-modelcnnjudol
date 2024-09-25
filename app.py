from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

app = Flask(__name__)

# Muat semua model
models = [
load_model('D:/WEBSITE/flaskjudolpredict/models/modelbaru8.h5') 
]

# Fungsi untuk mengunduh dan membaca gambar dari URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

# Fungsi untuk mengekstraksi URL gambar dari halaman web
def extract_image_urls(url):
    image_urls = []
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for img in soup.find_all('img'):
            img_url = img.get('src')
            if img_url:
                img_url = urljoin(url, img_url)
                parsed_url = urlparse(img_url)
                if parsed_url.scheme in ['http', 'https']:
                    image_urls.append(img_url)
    except Exception as e:
        print("Error:", e)
    return image_urls

# Fungsi untuk menampilkan gambar dan prediksi
def display_and_predict_images(image_urls, models):
    predictions = []
    
    for img_url in image_urls:
        try:
            img = load_image_from_url(img_url)
            img_array = np.array(img)
            
            resize = tf.image.resize(img_array, (256, 256))
            img_expanded = np.expand_dims(resize / 255.0, axis=0)
            
            model_predictions = []
            for i, model in enumerate(models):
                yhat = model.predict(img_expanded)
                class_name = 'Memprediksi Bukan Judi' if yhat > 0.5 else 'Memprediksi Terindikasi Judi'
                model_predictions.append({'model': i + 1, 'class': class_name})
            
            predictions.append({'url': img_url, 'predictions': model_predictions})
            
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")

        # Sort predictions to have 'Judi' class images first
    predictions.sort(key=lambda x: x['predictions'][0]['class'] != 'Memprediksi Terindikasi Judi')    

    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    website_url = data.get('url')

    if not website_url:
        return jsonify({'error': 'URL is required'}), 400

    image_urls = extract_image_urls(website_url)
    if not image_urls:
        return jsonify({'error': 'No images found on the provided URL'}), 404

    predictions = display_and_predict_images(image_urls, models)
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
