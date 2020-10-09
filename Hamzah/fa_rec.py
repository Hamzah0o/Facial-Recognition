import numpy as np
import cv2
import pickle



# using haar cascade
face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
lables = {"person name": 1}
with open("lables.pickle",'rb') as f:
    og_lables = pickle.load(f)
    lables = {v:k for k,v in og_lables.items()}
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to gray
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y , w , h) in faces :
        #print(x, y , w , h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #recognition deep learned model
        id_, conf =recognizer.predict(roi_gray)
        if conf>=50 and conf<=100:
            print(conf)
            print(id_)
            print(lables[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = lables[id_]
            color= (255, 255, 255)
            stroke= 2
            cv2.putText(frame,name, (x,y), font, 1 , color, stroke, cv2.LINE_AA)


        img_item = "my-image.png"
        cv2.imwrite(img_item , roi_gray)
        color = (255, 106, 0)
        stroke = 2
        end_cord_x= x+w
        end_cord_y= y+h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
        #eyes = eye_cascade.detectMultiScale(roi_color)
        #for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
