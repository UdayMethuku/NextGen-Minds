import cv2
import numpy as np
from openvino import Core
import gradio as gr

# Load OpenVINO models
core = Core()
face = core.compile_model("intel/face-detection-0200/FP32/face-detection-0200.xml", "CPU")
emotion = core.compile_model("intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml", "CPU")

face_in = face.input(0)
face_out = face.output(0)
emotion_in = emotion.input(0)
emotion_out = emotion.output(0)

emotion_labels = ["neutral", "happy", "sad", "surprise", "anger"]

def detect_emotion(frame):
    h, w = frame.shape[:2]
    face_img = cv2.resize(frame, (face_in.shape[3], face_in.shape[2]))
    face_img = face_img.transpose((2, 0, 1))[np.newaxis, ...]

    dets = face([face_img])[face_out]
    for det in dets[0][0]:
        conf = det[2]
        if conf > 0.6:
            x1, y1, x2, y2 = map(lambda v: int(v * [w, h, w, h][dets[0][0].tolist().index(det)]), det[3:7])
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            em_img = cv2.resize(face_crop, (64, 64)).transpose((2, 0, 1))[np.newaxis, ...]
            res = emotion([em_img])[emotion_out]
            label = emotion_labels[np.argmax(res)]
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame

iface = gr.Interface(fn=detect_emotion, inputs=gr.Image(source="webcam", streaming=True), outputs="image")
iface.launch(share=True)
