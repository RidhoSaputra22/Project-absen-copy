from detect import *
from datetime import *
from flask import Flask, render_template, Response, jsonify
import cv2
import pandas as pd
import time

app = Flask(__name__)

required_shape = (160,160)
face_encoder = InceptionResNetV2()
path_m = "facenet_keras_weights.h5"
face_encoder.load_weights(path_m)
encodings_path = 'encodings/encodings.pkl'
face_detector = mtcnn.MTCNN()
encoding_dict = load_pickle(encodings_path)

cap = cv2.VideoCapture(0)

df_fr = pd.read_csv('./database/fr.csv')
df_amd = pd.read_csv('./database/amd.csv')

this_time = datetime.now().strftime("%H:%M:%S")

name = "unknown"


#ambil data wajah
def generate_frames():
    global name
    while cap.isOpened():
        ret,frame = cap.read()

        if not ret:
            print("CAM NOT OPEND") 
            break

        # frame = detect(frame , face_detector , face_encoder , encoding_dict)
        print("start")

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.detect_faces(img_rgb)
        for res in results:
            if res['confidence'] < confidence_t:
                continue
            face, pt_1, pt_2 = get_face(img_rgb, res['box'])
            encode = get_encode(face_encoder, face, required_size)
            encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
            name = 'unknown'

            distance = float("inf")
            for db_name, db_encode in encoding_dict.items():
                dist = cosine(db_encode, encode)
                if dist < recognition_t and dist < distance:
                    name = db_name
                    distance = dist

            if name == 'unknown':
                cv2.rectangle(frame, pt_1, pt_2, (0, 0, 255), 2)
                cv2.putText(frame, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            else:
                print(name)
                cv2.rectangle(frame, pt_1, pt_2, (0, 255, 0), 2)
                cv2.putText(frame, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 200, 200), 2)
                df_fr.loc[len(df_fr.index)] = [name, this_time, date.today()]
                df_fr.to_csv('database/fr.csv', index=False)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
def generate_data():
    global name
    ridho_row = df_amd.loc[df_amd['Nama'] == str(name)]
    ridho_row_str = ridho_row.to_csv(header=False, index=False).strip()
    while True:
        yield f"data: {ridho_row_str} + ={name}=\n\n"
        time.sleep(1)

#untuk menampikan data di interface        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/updates')
def sse():
    return Response(generate_data(), content_type='text/event-stream')

if __name__ == '__main__':
    # print(name)
    app.run(debug=True)
