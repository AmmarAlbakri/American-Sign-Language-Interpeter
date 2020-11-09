import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import cv2
import numpy as np
from keras.utils import np_utils
import os
import sys

class_count = {'A': 0, 'B': 0, 'C': 0, "D": 0,
               'E': 0, "F": 0, "G": 0, "H": 0, "I": 0, 'J': 0, "K": 0,
               "L": 0, "M": 0, "N": 0, "O": 0, "P": 0, "Q": 0, "R": 0, "S": 0, "T": 0,
               "U": 0, "V": 0, "W": 0, "X": 0, "Y": 0, 'Z': 0, 'nothing': 0, 'space': 0}

class_encoder = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, "O": 14, "P": 15, "Q": 16,
    "R": 17, "S": 18, "T": 19,
    "U": 20, "V": 21, "W": 22, "X": 23, "Y": 24, 'Z': 25, 'nothing': 26, 'space': 27
}

path = "Data"
files = os.listdir(path)

X_train = []
Y_train = []
X_val = []
Y_val = []
X_test = []
Y_test = []

print("Loading files...")
for file_name in files:
    print(file_name, " ", end="")
    sys.stdout.flush()
    label = file_name
    myPicList = os.listdir(path + '/' + str(file_name))
    for image_name in myPicList:
        imgPath = path + '/' + str(file_name) + '/' + str(image_name)
        image = cv2.imread(imgPath)
        resized_image = cv2.resize(image, (64, 64))
        if class_count[label] < 2300:
            class_count[label] += 1
            X_train.append(resized_image)
            Y_train.append(class_encoder[label])
        elif 2300 <= class_count[label] < 2900:
            class_count[label] += 1
            X_val.append(resized_image)
            Y_val.append(class_encoder[label])
        elif 2900 <= class_count[label] < 3000:
            class_count[label] += 1
            X_test.append(resized_image)
            Y_test.append(class_encoder[label])

Y_train = np_utils.to_categorical(Y_train)
Y_val = np_utils.to_categorical(Y_val)
Y_test = np_utils.to_categorical(Y_test)

npy_data_path = "Numpy_Full"

np.save(npy_data_path + '/train_set.npy', X_train)
np.save(npy_data_path + '/train_classes.npy', Y_train)

np.save(npy_data_path + '/validation_set.npy', X_val)
np.save(npy_data_path + '/validation_classes.npy', Y_val)

np.save(npy_data_path + '/test_set.npy', X_test)
np.save(npy_data_path + '/test_classes.npy', Y_test)

print("Data pre-processing Success!")
