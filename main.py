import cv2
import numpy as np
import face_recognition


imgOsama = face_recognition.load_image_file('ImagesBasic/Images5.jpg')
imgOsama = cv2.cvtColor(imgOsama, cv2.COLOR_BGR2RGB)

imgOsama1 = face_recognition.load_image_file('ImagesBasic/Image2.jpg')
imgOsama1 = cv2.cvtColor(imgOsama1, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgOsama)[0]
encodeOsama = face_recognition.face_encodings(imgOsama)[0]

faceloc2 = face_recognition.face_locations(imgOsama1)[0]
encodeOsama2 = face_recognition.face_encodings(imgOsama1)[0]

cv2.rectangle(imgOsama, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255, 0, 255, 0), 2)
cv2.rectangle(imgOsama1, (faceloc2[3], faceloc2[0]), (faceloc2[1], faceloc2[2]), (255, 0, 255, 0), 2)

results = face_recognition.compare_faces([encodeOsama], encodeOsama2)
faceDis = face_recognition.face_distance([encodeOsama], encodeOsama2)
cv2.putText(imgOsama1, f'{results}{round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

print("Image match status", results, faceDis)
print(faceloc)
print(faceloc2)
# cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE)
imgS = cv2.resize(imgOsama, (800, 600))
img1s = cv2.resize(imgOsama1, (800, 600))

cv2.imshow('Sir Osama', imgS)
cv2.imshow('Uzair', img1s)

cv2.waitKey(0)
