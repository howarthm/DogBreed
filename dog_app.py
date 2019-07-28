#!/usr/bin/env python
# coding: utf-8

from sklearn.datasets import load_files       
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import model_from_json
from keras.applications.resnet50 import preprocess_input, decode_predictions
from extract_bottleneck_features import *                
from tqdm import tqdm
import numpy as np
from glob import glob
import cv2

class dog_classifier:
    def __init__(self, algorithm_path, weight_path):
        # load list of dog names
        self.dog_names = [item[20:-1] for item in sorted(glob("data/dogImages/train/*/"))]
        # load model from saved values from notebook
        with open(algorithm_path, 'r') as f:#with open('saved_models/resnet50.json', 'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(weight_path)#self.model.load_weights('saved_models/weights.best.Resnet50.hdf5')
        # extract pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
        # define ResNet50 model
        self.ResNet50_model = ResNet50(weights='imagenet')

    # returns "True" if face is detected in image stored at img_path
    def face_detector(self, img_path):
        """Detect if the picture is a human face.  Takes an image path as its argument.
        Returns True is a face is detected.
        """
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def path_to_tensor(self, img_path):
        """Returns tensor of image
        """
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def ResNet50_predict_labels(self, img_path):
        """returns prediction vector for image located at img_path
        """
        img = preprocess_input(self.path_to_tensor(img_path))
        return np.argmax(self.ResNet50_model.predict(img))

    def dog_detector(self, img_path):
        """Return True is a dog is detected in the image
        """
        prediction = self.ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151)) 

    # Thanks to this answer which helped me with the latest version of keras 
    # https://stackoverflow.com/questions/51231576/tensorflow-keras-expected-global-average-pooling2d-1-input-to-have-shape-1-1
    def Resnet50_predict_breed(self, img_path):
        """Returns predicted breed from img_path
        """
        # extract bottleneck features
        bottleneck_feature = extract_Resnet50(self.path_to_tensor(img_path))
        bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
        bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
        # obtain predicted vector
        predicted_vector = self.model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return self.dog_names[np.argmax(predicted_vector)]

    def dog_name(self, dog_name_path):
        """Returns a clean version of the dog name
        """
        return dog_name_path.split('.')[-1].replace('_', ' ')
    
    def classify_dog_breed(self, image_path):
        """Returns text specifying the breed or an error if the picture is not a dog or human.
        """
        if self.dog_detector(image_path):
            return 'This is a dog, and this photo looks like a '+self.dog_name(self.Resnet50_predict_breed(image_path))
        elif self.face_detector(image_path):
            return 'This is a human, and if this person were a dog, they would resemble a '+self.dog_name(self.Resnet50_predict_breed(image_path))
        else:
            return 'Error: not a dog or human'
        
#classifier = dog_classifier('saved_models/resnet50.json','saved_models/weights.best.Resnet50.hdf5')

# If Tom Selleck were a dog, what would he be?
#print(classifier.classify_dog_breed('images/tomselleck.jpg'))




