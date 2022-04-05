from flask import Flask, Response, render_template
import cv2
import tensorflow_hub as hub
import numpy
import pandas as pd
import tensorflow as tf

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

            if phone or persons != 1:
                cheatingframe += 1
            framecount += 1

            cv2.putText(inp, f'Persons: {persons}, Phone: {phone}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                        2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', inp)
            # frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            break


@app.route('/')
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
