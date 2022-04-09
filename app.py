from flask import Flask, Response, render_template
import cv2
import tensorflow_hub as hub
import numpy
import pandas as pd
import tensorflow as tf
import mediapipe as mp
import time
from mediapipe.framework.formats import landmark_pb2
import matplotlib


class FaceMeshDetector():

    def __init__(self,staticMode = False,maxFaces =2,minDetectionConfidence = .7,minTrackConfidence = 0.7):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackConfidence = minTrackConfidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode= self.staticMode,
                                                 max_num_faces=self.maxFaces,
                                                 min_detection_confidence=self.minDetectionConfidence,
                                                 min_tracking_confidence=self.minTrackConfidence
                                                 )
        self.drawSpecs = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)


    def findFaceMesh(self,img,points,draw,mouthOpenAnalysis,upperLipsPoints,lowerLipsPoints,centerLipsPoints):

        # Convert Image to RGB in order to analyse
        self.imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # Find the Faces and the Landmarks
        self.results = self.faceMesh.process(self.imgRGB)

        # Stores information about landmarks (x,y,z) for each face
        faces = []
        mouthState = False

        if(self.results.multi_face_landmarks):
            # Iterating over landmarks for all faces
            for face_landmarks in self.results.multi_face_landmarks:

                # Used for display ( display only the concerned points)
                landmark_subset = landmark_pb2.NormalizedLandmarkList(
                    landmark=[
                        face_landmarks.landmark[x] for x in points
                    ]
                )



                if draw==True :
                    # Draw the landmarks, passing the landmark_subset and the landmark_list
                    self.mpDraw.draw_landmarks(img,
                                               landmark_list=landmark_subset,
                                               landmark_drawing_spec = self.drawSpecs)


                if(mouthOpenAnalysis):
                    # This function returns True or False if mouth is open or closed, takes in the lips coordinates
                    # face_landmarks, the image, and a ratio

                    result = self.mouthOpen(upperLipsPoints,lowerLipsPoints,
                                            centerLipsPoints,
                                            face_landmarks.landmark,img,0.7)

                    if(result):
                        mouthState = True
                        # cv2.putText(img,"MOUTH OPEN",(20,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
                    else:
                        pass
                        # cv2.putText(img, "MOUTH CLOSED", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                # List of landmarks for the face
                face = []
                for id,lm in enumerate(face_landmarks.landmark):
                    ih,iw,iz = img.shape
                    # Undoing the standardization
                    x,y,z = int(lm.x*iw),int(lm.y*ih),int(lm.z*iz)

                    # Only append if landmark is of our concern
                    if(id in points):
                        # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        face.append([id,x,y,z])
                faces.append(face)

        return mouthState,img,faces

    def mouthOpen(self,upperLipsPoints,lowerLipsPoints,centerLipsPoints,landmarks,img,ratio):

        upperLipsDistance = 0
        LowerLipsDistance = 0
        centerLipsDistance = 0

        # These loops, calculate the distance for upper/lower lips and the center
        # Simple euclidean distance has been used

        for i in range(len(upperLipsPoints)):

            p1 = upperLipsPoints[i][0]
            p2 = upperLipsPoints[i][1]

            lm_1 = landmarks[p1]
            lm_2 = landmarks[p2]

            ih, iw, iz = img.shape
            x_1, y_1, z_1 = int(lm_1.x * iw), int(lm_1.y * ih), int(lm_1.z * iz)
            x_2, y_2, z_2 = int(lm_2.x * iw), int(lm_2.y * ih), int(lm_2.z * iz)

            dist = ((x_1-x_2)**2 + (y_1-y_2)**2 + (z_1-z_2)**2)**(1/3)

            upperLipsDistance += (dist/len(upperLipsPoints))

        for i in range(len(lowerLipsPoints)):

            p1 = lowerLipsPoints[i][0]
            p2 = lowerLipsPoints[i][1]

            lm_1 = landmarks[p1]
            lm_2 = landmarks[p2]

            ih, iw, iz = img.shape
            x_1, y_1, z_1 = int(lm_1.x * iw), int(lm_1.y * ih), int(lm_1.z * iz)
            x_2, y_2, z_2 = int(lm_2.x * iw), int(lm_2.y * ih), int(lm_2.z * iz)

            dist = ((x_1-x_2)**2 + (y_1-y_2)**2 + (z_1-z_2)**2)**(1/3)

            LowerLipsDistance += (dist/len(lowerLipsPoints))

        for i in range(len(lowerLipsPoints)):

            p1 = centerLipsPoints[i][0]
            p2 = centerLipsPoints[i][1]

            lm_1 = landmarks[p1]
            lm_2 = landmarks[p2]

            ih, iw, iz = img.shape
            x_1, y_1, z_1 = int(lm_1.x * iw), int(lm_1.y * ih), int(lm_1.z * iz)
            x_2, y_2, z_2 = int(lm_2.x * iw), int(lm_2.y * ih), int(lm_2.z * iz)

            dist = ((x_1-x_2)**2 + (y_1-y_2)**2 + (z_1-z_2)**2)**(1/3)

            centerLipsDistance += (dist/len(centerLipsPoints))


        if centerLipsDistance > min(upperLipsDistance, LowerLipsDistance) * ratio:
            return True
        else:
            return False


upperLips = [
    [0,12],
    [37,38],
    [267,268]
]

lowerLips = [
    [15,17],
    [84,86],
    [314,316]
]

centerRegion = [
    [13,14],
    [82,87],
    [312,317]

]


# This if for displaying, contains the union of the upperLips, lowerLips, centerRegion

lipsPointsForAnalysis = [
    0,12,37,38,267,268,15,
    17,84,86,314,316,13,14,
    82,87,312,317
]

app = Flask(__name__)

video = cv2.VideoCapture(0)

model_url = 'https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1'
detector = hub.load(model_url)

framecount = 0
cheatingframe = 0





def gen_frames():
    global framecount
    global cheatingframe
    framecount = 0
    cheatingframe = 0

    lip_detector = FaceMeshDetector(maxFaces=1)

    while True:

        success, frame = video.read()
        if success:

            inp = cv2.resize(frame, (512, 512))
            rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
            rgb_tensor = tf.expand_dims(rgb_tensor, 0)
            boxes, scores, classes, num_detections = detector(rgb_tensor)
            scores = scores.numpy()[0]
            classes = classes.numpy().astype(int)[0]

            phone = False
            persons = 0
            for i in range(len(classes)):
                if scores[i] > 0.5:
                    if classes[i] == 1:
                        persons += 1
                    elif classes[i] == 77:
                        phone = True

            mouth_state,_, _ = lip_detector.findFaceMesh(inp, lipsPointsForAnalysis, False, True, upperLips, lowerLips,
                                               centerRegion)

            if phone or persons != 1 or mouth_state:
                cheatingframe += 1
            framecount += 1

            cv2.putText(inp, f'Persons: {persons} Phone: {phone} ', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                        2, cv2.LINE_AA)
            cv2.putText(inp, f'Lips: {"Open" if mouth_state else "Closed"}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                        2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', inp)
            # frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            break


@app.route('/', methods=['GET', 'POST'])
def index():
    framecount = 0
    cheatingframe = 0
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/endexam', methods=['GET', 'POST'])
def endexam():
    cheatingscore = cheatingframe / framecount * 100
    return render_template('endexam.html', score=cheatingscore)


if __name__ == "__main__":
    app.run(debug=True)
