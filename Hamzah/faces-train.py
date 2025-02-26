import os
import numpy as np
from PIL import Image
import  cv2
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
#facial recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id= 0
lable_ids = {}
y_lables = []
x_train = []

for root, dirs, files, in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            lable = os.path.basename(root).replace(" "," ").lower()
            print(path,lable)
            if not lable in lable_ids:
                lable_ids[lable] = current_id
                current_id += 1
            id_ = lable_ids[lable]
            #print(lable_ids)
            #y_lables.append(lable)
            #x_train.append(path)
            pill_image = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pill_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pill_image, "uint8")
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h , x:x+w]
                x_train.append(roi)
                y_lables.append(id_)

#print(y_lables)
#print(x_train)

with open("lables.pickle",'wb') as f:
    pickle.dump(lable_ids, f)
recognizer.train(x_train, np.array(y_lables))
recognizer.save("trainner.yml")