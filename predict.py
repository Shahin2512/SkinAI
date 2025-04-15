# predict.py
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

model = load_model("model/skin_disease_model.h5")
class_labels = os.listdir("data/Dataset/train")  # Folder names are class names

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    return class_labels[class_index]
