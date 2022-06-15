import cv2

#importing the faces haarcacscade algorithm
trained_face_data= cv2.CascadeClassifier('face.xml')

#read the image
webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read, frame = webcam.read()

    #convert to grey scale
    greyimg = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)


    #detect faces
    face_coordinates= trained_face_data.detectMultiScale(greyimg)
    
    #draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)



    #show the frame/ display the webcam
    cv2.imshow('face', frame)

    #wait/ show the image untill a key is pressed
    key= cv2.waitKey(1)

    if key== 113 or key== 81:
        break

webcam.release()

print("code completed")
