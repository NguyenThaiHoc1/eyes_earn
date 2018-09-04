import cv2
from PIL import Image, ImageDraw
import face_recognition as face_recognition
from imutils import face_utils


frame = cv2.imread("./images_test/obama.jpg")

cv2.imwrite("./images_test/hell2.jpg", frame)

rgb_frame = frame[:, :, ::-1]

cv2.imwrite("./images_test/hell1.jpg", rgb_frame)


# face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

# face_locations = face_recognition.face_locations(rgb_frame)

# for top, right, bottom, left in face_locations:
# 	cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

# 	cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)

# cv2.imwrite("./images_test/hello.jpg", frame)

# pil_image = Image.fromarray(frame)
# d = ImageDraw.Draw(pil_image)

# for face_landmarks in face_landmarks_list:

#     # Print the location of each facial feature in this image
#     for facial_feature in face_landmarks.keys():
#         print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

#     # Let's trace out each facial feature in the image with a line!
#     for facial_feature in face_landmarks.keys():
#         d.line(face_landmarks[facial_feature], width=5)



(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
print(lStart, lEnd)
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
print(rStart, rEnd)
rects = face_recognition.rect_face_location_raw(rgb_frame, 0)
for rect in rects:

		shape = face_recognition.rect_face_landmarks_raw(rgb_frame, rect)
		shape = face_utils.shape_to_np(shape)
		print(lStart,lEnd)
		print(rStart,rEnd)
		
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = face_recognition.eye_aspect_ratio(leftEye)
		rightEAR = face_recognition.eye_aspect_ratio(rightEye)
 
		ear = (leftEAR + rightEAR) / 2.0

		print (ear)

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


cv2.imwrite("./images_test/hello4.jpg", frame)



