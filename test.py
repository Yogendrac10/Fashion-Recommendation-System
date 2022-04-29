from fileinput import filename
import pickle
import numpy as np
import tensorflow 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2


#Getting embeddings and file names 

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
file_names = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
    ])


img = image.load_img('D:/Deep Learning Projects/Fashion recomender system/dataset/Sample/t-shirt.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocess_img = preprocess_input(expanded_img_array)
result = model.predict(preprocess_img).flatten()
normalized_result =  result / np.linalg.norm(result)  


# Finding Nearest neighbors

neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')

neighbors.fit(feature_list)

# Finding the top 5 neighbors
distances, indices = neighbors.kneighbors([normalized_result])

# accessing filenames from indices using[0] because this is 2D list
for file in indices[0]:
    temp_img = cv2.imread(file_names[file])
    cv2.imshow('outpur', cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)


    
