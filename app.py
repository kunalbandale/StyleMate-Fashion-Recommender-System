from flask import Flask, render_template, request
import os
import tensorflow as tf
from PIL import Image
import pickle
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D

app = Flask(__name__)

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.filename), "wb") as f:
            f.write(uploaded_file.read())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and save_uploaded_file(file):
            features = feature_extraction(os.path.join("uploads", file.filename), model)
            indices = recommend(features, feature_list)
            return render_template('result.html', images=[filenames[i] for i in indices[0]])
        else:
            return "Error occurred"
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
