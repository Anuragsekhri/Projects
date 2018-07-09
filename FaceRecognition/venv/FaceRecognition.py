import cv2
import numpy as np
import os


subjects=["","Shivam Gupta","Sachin"]


def detectFace(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)
    if(len(faces) == 0):
        return None,None

    print("Type of Faces ",type(faces))
    print(faces)
    (x,y,w,h) = faces[0];

    return  gray[y:y+w,x:x+h],faces[0]

def prepareData(path):

    dirs = os.listdir(path)

    faces=[]
    labels=[]

    for dir_name in dirs :
        if not dir_name.startswith("s"):
            continue;

        label = int(dir_name.replace("s",""))

        sub_path = path + "/" + dir_name

        sub_img_name = os.listdir(sub_path)

        for img_name in sub_img_name:

            if img_name.startswith("."):
                continue;

            image_path = sub_path+"/"+ img_name;

            #read image
            image = cv2.imread(image_path)

            cv2.imshow("Training Image .....",cv2.resize(image,(400,500)))
            cv2.waitKey(100)

            face,rect = detectFace(image)

            if face is not None:

                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces,labels



print("Preparing Data !!!!!!!!!")
#faces,labels = prepareData("training-data")
faces, labels = prepareData("training-data")
print("Data Prepared..........")

print("Total Faces : ",len(faces) )
print("Total labels : ",len(labels))


face_recogniser = cv2.face.LBPHFaceRecognizer_create()
face_recogniser.train(faces,np.array(labels))



def drawRect(img,rect):
    (x,y,w,h) = rect
    cv2.rectangle(img,(x,y) , (x+w,y+h),(0,255,0) , 2)


def drawText(img,text,x,y):
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),7)


def predict(test_img):
    img = test_img.copy()
    face,rect = detectFace(img)

    label,confidence = face_recogniser.predict(face)
    label_text = subjects[label]

    drawRect(img,rect)
    drawText(img,label_text,rect[0],rect[1]-5)


    return img



print("predicting !!!!!!!!!!!")

testImg1 = cv2.imread("test-data/test1.jpg")
testImg2 = cv2.imread("test-data/test2.jpg")


predictedImg1 = predict(testImg1)
predictedImg2 = predict(testImg2)

print("Prediction Done ")

cv2.imshow("Results",cv2.resize(predictedImg1,(400,500)))

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Results",cv2.resize(predictedImg2,(400,500)))

cv2.waitKey(0)
cv2.destroyAllWindows()

