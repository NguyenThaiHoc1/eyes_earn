import cv2
from PIL import Image, ImageDraw
import face_recognition as face_recognition
from imutils import face_utils
import imutils

EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 2
COUNTER = 0
TOTAL = 0
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # rgb_frame = frame[:, :, ::-1]
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = face_recognition.rect_face_location_raw(gray, 0)

    print (len(rects))
    for rect in rects:
    	shape = face_recognition.rect_face_landmarks_raw(gray, rect)
    	shape = face_utils.shape_to_np(shape)
    	leftEye = shape[lStart:lEnd]
    	rightEye = shape[rStart:rEnd]
    	leftEAR = face_recognition.eye_aspect_ratio(leftEye)
    	rightEAR = face_recognition.eye_aspect_ratio(rightEye)
    	ear = (leftEAR + rightEAR) / 2.0
    	leftEyeHull = cv2.convexHull(leftEye)
    	rightEyeHull = cv2.convexHull(rightEye)
    	cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    	cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    	if ear < EYE_AR_THRESH:
    		COUNTER += 1
    	else:
    		if COUNTER >= EYE_AR_CONSEC_FRAMES:
    			TOTAL += 1
    		COUNTER = 0
    	cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    	cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('frame', cv2.resize(frame, (500, 500)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break


cap.release()
cv2.destroyAllWindows()