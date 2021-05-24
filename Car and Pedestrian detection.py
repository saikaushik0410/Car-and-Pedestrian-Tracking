import cv2 

#img = cv2.imread('actual.jpg') 
video = cv2.VideoCapture('carsandped.mp4')
car_tracker = cv2.CascadeClassifier('haarcascade_car.xml')
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')


while True:
    frame_detect, frame = video.read()

    grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cars = car_tracker.detectMultiScale(grayscale)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscale)

    for(x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
    
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)

    cv2.imshow('car detector',frame)
    key = cv2.waitKey(2)

    if key == 81 or key == 113:
        break
video.release()
print("Code Completed")