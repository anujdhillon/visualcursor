import cv2
import numpy as np
import dlib
import math
import pyautogui
import csv
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
pyautogui.FAILSAFE = False
modelx = keras.models.load_model("modelx")
modely = keras.models.load_model("modely")

font = cv2.FONT_HERSHEY_COMPLEX
def mid(p1,p2):
    return (p1.x + p2.x)//2, (p1.y + p2.y)//2

def dis(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def getratio(l,r,t,b):
    return dis(t,b)/dis(l,r)

capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector() #face object
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #gets facial landmarks
while(True):
    _, frame = capture.read() #webcam capture
    h,w,_ = frame.shape
    grayframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(grayframe) #detects all faces in the webcam
    for face in faces:
        #x,y = face.left(), face.top()
        #x1,y1 = face.right(),face.bottom()
        #cv2.rectangle(frame,(x,y),(x1,y1),(0,255,0),1)
        landmarks = predictor(grayframe,face)

        left_left = (landmarks.part(36).x,landmarks.part(36).y) #left eye coordinates
        left_right = (landmarks.part(39).x,landmarks.part(39).y)
        left_top = mid(landmarks.part(37),landmarks.part(38))
        left_bottom = mid(landmarks.part(41), landmarks.part(40))
        left_eye_region = np.array(
            [(landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(37).x, landmarks.part(37).y),
             (landmarks.part(38).x, landmarks.part(38).y), (landmarks.part(39).x, landmarks.part(39).y),
             (landmarks.part(40).x, landmarks.part(40).y), (landmarks.part(41).x, landmarks.part(41).y), ])

        right_left = (landmarks.part(42).x, landmarks.part(42).y) #right eye coordinates
        right_right = (landmarks.part(45).x, landmarks.part(45).y)
        right_top = mid(landmarks.part(43), landmarks.part(44))
        right_bottom = mid(landmarks.part(47), landmarks.part(46))
        right_eye_region = np.array(
            [(landmarks.part(42).x, landmarks.part(42).y), (landmarks.part(43).x, landmarks.part(43).y),
             (landmarks.part(44).x, landmarks.part(44).y), (landmarks.part(45).x, landmarks.part(45).y),
             (landmarks.part(46).x, landmarks.part(46).y), (landmarks.part(47).x, landmarks.part(47).y), ])

        left_ratio = getratio(left_left,left_right,left_top,left_bottom) #ver/hor ratio
        right_ratio = getratio(right_left, right_right, right_top, right_bottom)
        avg_ratio = (left_ratio+right_ratio)/2

        l_min_x = np.min(left_eye_region[:,0]) + (left_top[1] - left_bottom[1])//2
        l_max_x = np.max(left_eye_region[:, 0]) - (left_top[1] - left_bottom[1])//2
        l_min_y = np.min(left_eye_region[:, 1]) + (left_top[1] - left_bottom[1])//2
        l_max_y = np.max(left_eye_region[:, 1]) - (left_top[1] - left_bottom[1])//2

        r_min_x = np.min(right_eye_region[:, 0]) + (right_top[1] - right_bottom[1])//2
        r_max_x = np.max(right_eye_region[:, 0]) - (right_top[1] - right_bottom[1])//2
        r_min_y = np.min(right_eye_region[:, 1]) + (right_top[1] - right_bottom[1])//2
        r_max_y = np.max(right_eye_region[:, 1]) - (right_top[1] - right_bottom[1])//2

        left_eye = cv2.resize(grayframe[l_min_y:l_max_y,l_min_x:l_max_x],(30,18))
        right_eye = cv2.resize(grayframe[r_min_y:r_max_y, r_min_x:r_max_x], (30, 18))
        eyes = np.concatenate((left_eye, right_eye), axis=1)
        eyes_data = (eyes.flatten())
        topredict = eyes_data.reshape((1,1080))
    x = modelx.predict(topredict)[0][0]
    y = modely.predict(topredict)[0][0]
    pyautogui.moveTo(x, y)
    cv2.imshow("mask", eyes)
    key = cv2.waitKey(10)
    if(key == 27):
        break
    elif key % 256 == 32:
        # SPACE pressed
        with open('training_data.csv', 'a+', newline='') as file:
            writer = csv.writer(file)
            x, y = pyautogui.position()
            label = 2003*x + y
            mousepos = np.asarray([label])
            datarow = np.concatenate((eyes_data, mousepos), axis=0)
            writer.writerow(datarow)


capture.release()
cv2.destroyAllWindows()
