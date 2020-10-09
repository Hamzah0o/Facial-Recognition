import os
import cv2


face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
path= 'D:\/My file\\Python\\Hamzah\\images'
os.chdir(path)
Newfolder= input("Enter your name ")
os.makedirs(Newfolder)
full_path=os.path.join(path, Newfolder)
print(full_path)
os.chdir(full_path)
cam = cv2.VideoCapture(0)
count=0

while True :
    ret, img= cam.read()
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1 , 4)
    for (x, y , w , h) in faces :
        cv2.rectangle(img, (x,y) , (x+w, y+h), (255 , 0 , 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    name = "Self in regestration system"
    color = (0, 223, 255)
    stroke = 2
    cv2.putText(img, name, (25,25), font, 1, color, stroke, cv2.LINE_AA)
    cv2.imshow("test",img)

    if not ret:
        break
    k=cv2.waitKey(1)

    if k%256==27:
        print("close")
        break
    elif k%256==32:

        print("Image"+str(count)+"saved")
        file=full_path+'img'+str(count)+'.jpg'
        cv2.imwrite(file,img)
        count+=1

cam.release
cv2.destroyAllWindows()


