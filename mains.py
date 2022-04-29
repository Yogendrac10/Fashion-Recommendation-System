# -*- coding: utf-8 -*-


import numpy as np
import os
import tensorflow 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm import tqdm
import pickle



model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
    ])

#print(model.summary())

def  extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocess_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocess_img).flatten()
    normalized_result =  result / np.linalg.norm(result)
    
    return normalized_result


filenames = []

for i in os.listdir('D:/Deep Learning Projects/Fashion recomender system/dataset/images'):
    filenames.append(os.path.join('D:/Deep Learning Projects/Fashion recomender system/dataset/images', i))
    

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))









