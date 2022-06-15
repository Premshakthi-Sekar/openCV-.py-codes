import cv2

#importing the faces haarcacscade algorithm
trained_face_data= cv2.CascadeClassifier('face.xml')

#read the image
img=cv2.imread('test_img_joeychand.jfif')

#convert to grey scale
greyimg = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

#detect faces
face_coordinates= trained_face_data.detectMultiScale(greyimg)

#draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)



#show the face/ display the image
cv2.imshow('face', img)

#wait/ show the image untill a key is pressed
cv2.waitKey()

print("code completed")
