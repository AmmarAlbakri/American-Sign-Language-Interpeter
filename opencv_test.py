import numpy as np
import cv2
from keras.models import model_from_json
from pynput import keyboard

########### PARAMETERS ##############
width = 640
height = 480
threshold = 0.95  # MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 0
class_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
              'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','nothing','space']
cap = cv2.VideoCapture(cameraNo)
cap.set(3, width)
cap.set(4, height)

with open('Weights_Full/model.json', 'r') as f:
    model = model_from_json(f.read())
    model.summary()
model.load_weights('Weights_Full/model_weights.h5')

text = ""

curr = ""
prev = ""
same = 0


#### PREPORCESSING FUNCTION
def preProcessing(img):
    img = img / 255
    return img


def on_press(key):
    global text
    if key == keyboard.Key.backspace:
        text = text[:-1]
    elif key == keyboard.Key.space:
        text += " "


listener = keyboard.Listener(on_press=on_press)
listener.start()  # start to listen on a separate thread

while True:
    success, imgOriginal = cap.read()
    cv2.rectangle(imgOriginal, (100, 100), (300, 350), (0, 0, 255), 1)
    img = imgOriginal[100:350, 100:300]
    img = np.asarray(img)
    img = cv2.resize(img, (64, 64))
    cv2.imshow("Processsed Image", img)
    img = img.reshape(1, 64, 64, 3)
    #### PREDICT
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    probVal = np.amax(predictions)
    letter = class_list[classIndex]

    if probVal > threshold and letter != 'nothing':

        cv2.putText(imgOriginal, str(letter),
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 255, 255), 1)
        curr = letter
        if prev == "":
            prev = curr
            text += letter
        elif prev == curr:
            same += 1
        if same == 20:
            if letter == 'space':
                text+="_"
                same = 0
            else:
                print(letter, probVal, " same:", same)
                same=0
                text += letter
                text = text.replace("_"," ")
        prev = curr

    cv2.putText(imgOriginal, str(text),
                (150, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 0, 255), 1)
    cv2.imshow("Original Image", imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
