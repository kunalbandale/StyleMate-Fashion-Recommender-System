# import pickle
import pickle
import numpy as np
import tensorflow
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
# for debugging error
# img = image.load_img('sample/jursey.jpg')
# img_size = img.size
# print(img_size)

img = image.load_img('.\sample\watch.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# we can also try
# neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='manhattan')


neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])
print(indices)

# 2d list file number se file pick karna hai

for file in indices[0]:
    # print(filenames[file])
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img, (300,300)))
    cv2.waitKey(0)
# jo nearest images aye hai unko display karnenge ab
