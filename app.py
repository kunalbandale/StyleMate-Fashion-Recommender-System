# Import necessary libraries
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import os
from tqdm import tqdm
import pickle

# Load pre-trained ResNet50 model
# Set `include_top` to False to exclude the fully-connected layers at the top
# Set `input_shape` to (224, 224, 3) to match the expected input size of ResNet50
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the weights of the pre-trained ResNet50 model
model.trainable = False

# Create a Sequential model
model = tf.keras.Sequential([
    model,  # Add the pre-trained ResNet50 model as a layer
    GlobalMaxPooling2D()  # Add GlobalMaxPooling2D layer to obtain image embeddings
])


# print(model.summary())

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

# print(len(filenames))
# print(filenames[0:5])

# feature_list = [[],[],[],[]]
feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

print(np.array(feature_list).shape)

pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))

# feature extraction completed