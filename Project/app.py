from flask import Flask, render_template, Response
import os
import cv2
import numpy as np
from keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__)

def video_gen():
    global camera
    camera = cv2.VideoCapture(0)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.h5')
    model = tf.keras.models.load_model(model_path, compile=False)
    name = ["Found Missing","Normal"]
    while True:
        success, frame = camera.read()
        re_frame = cv2.resize(frame, (64,64))
        x  = image.img_to_array(re_frame)
        x = np.expand_dims(x,axis = 0)
        p = (model.predict(x) > 0.5).astype('int32')
        # print(p)
        cv2.putText(frame, "Predicted  Class = "+str(name[p[0][0]]), (100,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
        
        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
              b'<h2>Prediction: ' + name[p[0][0]].encode() + b'</h2>\r\n')

@app.route('/search', methods=['POST'])
def search():
    return render_template('search.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()