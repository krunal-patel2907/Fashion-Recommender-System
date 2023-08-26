import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

features_list=pickle.load(open('embeddings.pkl','rb'))
filenames=pickle.load(open('filenames.pkl','rb'))

print(features_list)