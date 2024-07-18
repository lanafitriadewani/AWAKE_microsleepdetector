from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        print('call')
        s = 'espeak "'+msg+'"'
        os.system(s)

    if alarm_status2:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 30
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

drowsy_start_time = None
drowsy_duration = 0
drowsy_count = 0
yawn_count = 0

drowsy_detected = False
yawn_detected = False

alert_level = ""

print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    # Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            if drowsy_start_time is None:
                drowsy_start_time = time.time()
                drowsy_count += 1
            drowsy_duration = time.time() - drowsy_start_time
            COUNTER += 1
            drowsy_detected = True

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if drowsy_duration > 6 or drowsy_count > 7:
                    alert_level = "ALERT LEVEL 4"
                elif drowsy_duration > 4 or drowsy_count > 5:
                    alert_level = "ALERT LEVEL 3"
                elif drowsy_duration > 3 or drowsy_count > 3:
                    alert_level = "ALERT LEVEL 2"
                elif drowsy_duration > 2:
                    alert_level = "ALERT LEVEL 1"
                # else:
                #     alert_level = "DROWSINESS ALERT!"

                if not alarm_status:
                    alarm_status = True
                    t = Thread(target=alarm, args=('wake up sir',))
                    t.deamon = True
                    t.start()
        else:
            COUNTER = 0
            drowsy_start_time = None
            drowsy_duration = 0
            drowsy_detected = False
            alarm_status = False

        if distance > YAWN_THRESH:
            if not yawn_detected:
                yawn_count += 1
                yawn_detected = True

            if yawn_count > 8:
                alert_level = "ALERT LEVEL 4"
            elif yawn_count > 6:
                alert_level = "ALERT LEVEL 3"
            elif yawn_count > 4:
                alert_level = "ALERT LEVEL 2"
            elif yawn_count > 2:
                alert_level = "ALERT LEVEL 1"
            # else:
            #     alert_level = "Yawn Alert"

            if not alarm_status2 and not saying:
                alarm_status2 = True
                t = Thread(target=alarm, args=('take some fresh air sir',))
                t.deamon = True
                t.start()
        else:
            yawn_detected = False
            alarm_status2 = False

        cv2.putText(frame, "EYE: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if drowsy_detected:
        cv2.putText(frame, "Drowsy Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if yawn_detected:
        cv2.putText(frame, "Yawn Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, "Drowsy Count: {}".format(drowsy_count), (10, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Yawn Count: {}".format(yawn_count), (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Drowsy Duration: {:.2f} sec".format(drowsy_duration), (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, alert_level, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == 13:  # Enter key pressed
        alert_level = ""
        drowsy_start_time = None
        drowsy_duration = 0
        drowsy_count = 0
        yawn_count = 0
        drowsy_detected = False
        yawn_detected = False

cv2.destroyAllWindows()
vs.stop()