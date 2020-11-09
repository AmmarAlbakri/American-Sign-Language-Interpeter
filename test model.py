import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json
from sklearn.metrics import accuracy_score

# dimensions of our images
image_size = 64

with open('Weights_Full/model.json', 'r') as f:
    model = model_from_json(f.read())
    model.summary()
model.load_weights('Weights_Full/model_weights.h5')

X_test=np.load("Numpy_Full/test_set.npy")
Y_test=np.load("Numpy_Full/test_classes.npy")
Y_predict = model.predict(X_test)

Y_predict = [np.argmax(r) for r in Y_predict]
Y_test = [np.argmax(r) for r in Y_test]
print("Predicts:",Y_predict)
print("Tests:",Y_test)

print("##################")
acc_score = accuracy_score(Y_test, Y_predict)
print("Accuracy: "+str(acc_score))
print("##################")
