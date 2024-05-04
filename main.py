import streamlit as st
import os
import tensorflow
from PIL import Image
# import pickle
import pickle
import numpy as np
import cv2
# from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D


feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
# print(feature_list)
filenames = pickle.load(open('filenames.pkl', 'rb'))


model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the weights of the pre-trained ResNet50 model
model.trainable = False

# Create a Sequential model
model = tensorflow.keras.Sequential([
    model,  # Add the pre-trained ResNet50 model as a layer
    GlobalMaxPooling2D()  # Add GlobalMaxPooling2D layer to obtain image embeddings
])



# StyleMate : Fahion Recommender System
st.title('StyleMate : Fashion Recommender System')


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# feature extraction
def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


# # -- steps --
# Upload file -> Save.
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        st.image(Image.open(uploaded_file))
        features = feature_extraction(os.path.join("uploads", uploaded_file.name),model)
        # st.text(features)
        indices = recommend(features,feature_list)
        #show
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])

    else:
        st.header('Error occurred')


# Load file -> Extract features.
# Recommendation.
# Show.
