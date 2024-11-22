from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
#development agar tidak menyimpan chache html/css
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response

app.config['TEMPLATES_AUTO_RELOAD'] = True
# Muat model VGG16 yang sudah dilatih
#model = load_model('./model/vgg16_spinach_detector_v001.h5')  # Sesuaikan path ke model Anda

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisasi gambar
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/klasifikasi')
def klasifikasi():
    return render_template('klasifikasi.html')

@app.route('/monitoring')
def monitoring():
    return render_template('monitoring.html')

@app.route('/bantuan')
def bantuan():
    return render_template('bantuan.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == "__main__":
    app.run(debug=True)
