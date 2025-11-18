import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0) 

framecount = 0

if not cam.isOpened(): 
    print("Error: Camera cannot be accessed")
    exit()

def detect_face(frame): 
    convert_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(convert_gray, 1.3, 5)

    if len(faces) == 0: 
        return frame, []
    
    # display face tracking - used for testing 
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return frame, faces


while True: 

    cond, img = cam.read()

    img, faces = detect_face(img)

    if len(faces) > 0: 
        x, y, w, h = faces[0]
        crop_img = img[y:y+h, x:x+w]

        if framecount < 10: 
            cv2.imwrite("C:/Users/Samko/OneDrive/Documents/School/deepfakeproj/images/image{}.png".format(framecount), crop_img)
            framecount += 1

    cv2.imshow('Video Face Detection', img)
 
    #exit program hotkeys
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

    status = cv2.getWindowProperty('Video Face Detection', cv2.WND_PROP_VISIBLE)

    if status == 0: 
        break

# Release resources
cam.release()
cv2.destroyAllWindows()