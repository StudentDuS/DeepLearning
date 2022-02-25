import cv2
import keras
import numpy as np
import time

model = keras.models.load_model('results/resnet50garbageClassificationTrain.h5')
cap = cv2.VideoCapture(0)
classes = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'healthProducts', 'metal', 'paper', 'plastic',
           'shoes', 'toothBrush', 'trash']
garbageClasses = {
    'battery': '有害垃圾(hazardousWaste)',
    'biological': '厨余垃圾(foodWaste)',
    'cardboard': '可回收垃圾(Recyclable)',
    'clothes': '可回收垃圾(Recyclable)',
    'glass': '可回收垃圾(Recyclable)',
    'healthProducts': '有害垃圾(hazardousWaste)',
    'metal': '可回收垃圾(Recyclable)',
    'paper': '可回收垃圾(Recyclable)',
    'plastic': '可回收垃圾(Recyclable)',
    'shoes': '可回收垃圾(Recyclable)',
    'toothBrush': '可回收垃圾(Recyclable)',
    'trash': '其他垃圾(residualWaste)',

}
while True:
    ret, fram = cap.read()
    if ret:
        fram = cv2.resize(fram, (224, 224))
        cv2.imshow('video', fram)
    else:
        break
    if cv2.waitKey(10) == ord('q'):
        img = cv2.resize(fram, (224, 224))
        img = np.reshape(img, [1, 224, 224, 3])
        p = model.predict(img)
        y = np.argmax(p)
        print(classes[y], garbageClasses[classes[y]], p)
        time.sleep(2)
