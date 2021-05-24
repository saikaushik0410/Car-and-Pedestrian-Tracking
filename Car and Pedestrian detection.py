import cv2 

#input video file
video = cv2.VideoCapture('carsandped.mp4')

#using pre-trained classifer files for cars and pedestrians
car_tracker = cv2.CascadeClassifier('haarcascade_car.xml')
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')


while True:

    #to take each frame from the input video
    frame_detect, frame = video.read()

    #Convert each frame of the video to grayscale
    grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detecting cars
    cars = car_tracker.detectMultiScale(grayscale)
    
    #detecting pedestrians
    pedestrians = pedestrian_tracker.detectMultiScale(grayscale)

    #draw rectangles around detected cars
    for(x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
    
    #draw rectangles around detected pedestrians
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)

    #to display the output
    cv2.imshow('car detector',frame)
    
    #to insert one millisecond delay in the output 
    key = cv2.waitKey(1)

    #Once the output is generated , q or Q can be used to terminate the video
    #ASCII value of  q = 113 and Q - 81
    if key == 81 or key == 113:
        break
#clean the data
video.release()

print("Code Completed")