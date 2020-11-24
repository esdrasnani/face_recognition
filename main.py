import cv2, os, numpy

dataset = ["", "Cesar", "Vini Correa", "Esdras", "Gustavo", "Vini Martins"]

def detect_faces(img) :
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = faceCasc.detectMultiScale(gray, 1.3, 5)
    graylist = []
    faceslist = []

    if len(faces) == 0 :
        return None, None

    for i in range(0, len(faces)) :
        (x, y, w, h) = faces[i]
        graylist.append(gray[y:y+w, x:x+h])
        faceslist.append(faces[i])

    return graylist, faceslist

def detect_face(img) :
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = faceCasc.detectMultiScale(gray, 1.3, 5)
    graylist = []
    faceslist = []

    if len(faces) == 0 :
        return None, None

    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def data() :
    dirs = os.listdir("Dataset")

    faces = []
    labels = []

    for i in dirs :
        set = "Dataset/" + i

        label = int(i)

        for j in os.listdir(set) :
            path = set + "/" + j
            img = cv2.imread(path)
            face, rect = detect_face(img)

            if face is not None :
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels

faces, labels = data()

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces, numpy.array(labels))

def predict(img) :

    face, rect = detect_faces(img)

    if face is not None :
        for i in range(0, len(face)) :
            label, confidence = face_recognizer.predict(face[i])
            print(confidence)
            if confidence < 100:
                label_text = dataset[label]
                color = (0, 255, 0);
                (x, y, w, h) = rect[i]
                cv2.line(img, (x, y), (int(x + (w/5)),y), color, 2)
                # line 2 : top right corner horizontal line 
                cv2.line(img, (int(x+((w/5)*4)), y), (x+w, y), color, 2)
                # line 3 : top left corner vertical line 
                cv2.line(img, (x, y), (x,int(y+(h/5))), color, 2)
                # line 4 : top right corner vertical line 
                cv2.line(img, (x+w, y), (x+w, int(y+(h/5))), color, 2)
                # line 5 : bottom left corner vertical line 
                cv2.line(img, (x, int(y+(h/5*4))), (x, y+h), color, 2)
                # line 6 : bottom left corner horizontal line 
                cv2.line(img, (x, int(y+h)), (x + int(w/5) ,y+h), color,2)
                # line 6 : bottom right corner horizontal line 
                cv2.line(img,(x+int((w/5)*4),y+h),(x + w, y + h),color, 2)
                # line 6 : bottom right corner verticals line 
                cv2.line(img, (x+w, int(y+(h/5*4))), (x+w, y+h), color, 2)


                pt1 = (int(x + w/2.0 -150), int(y+h+15))
                pt2 = (int((x + w/2.0 +50)+90), int(y+h+40))    
                pt3 = (int(x + w/2.0 -120), int(y+h +(-int(y+h) + int(y+h+20))/2+20))
                
                cv2.rectangle(img ,pt1, pt2, (255,255,255), -1)          
                cv2.rectangle(img, pt1, pt2, (0,0,255), 1) 
                cv2.putText(img, (label_text + "{:10.2f}".format((100-confidence)) + "%"),pt3 , cv2.FONT_HERSHEY_PLAIN, 1.1, (0,0,255))  

    return img

video_capture = cv2.VideoCapture(0)

'''
while True :
    ret, frame = video_capture.read()

    frame = predict(frame)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
'''
img = cv2.imread('Dataset/5/3.jpeg')
img = predict(img)
cv2.imshow('Image', img)
cv2.waitKey(0)
