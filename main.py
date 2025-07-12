import cv2
import numpy as np
from openvino import Core
import gradio as gr

# Load models
core = Core()
face = core.compile_model("intel/face-detection-0200/FP32/face-detection-0200.xml", "CPU")
emotion = core.compile_model("intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml", "CPU")

face_in, face_out = face.input(0), face.output(0)
emotion_in, emotion_out = emotion.input(0), emotion.output(0)
labels = ["neutral", "happy", "sad", "surprise", "anger"]

def detect_emotion(image):
    h, w = image.shape[:2]
    resized = cv2.resize(image, (face_in.shape[3], face_in.shape[2]))
    inp = np.expand_dims(resized.transpose(2, 0, 1), 0)
    faces = face([inp])[face_out]

    for det in faces[0][0]:
        if det[2] > 0.6:
            x1 = int(det[3] * w)
            y1 = int(det[4] * h)
            x2 = int(det[5] * w)
            y2 = int(det[6] * h)

            face_crop = image[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            emo_img = cv2.resize(face_crop, (64, 64)).transpose(2, 0, 1)
            emo_img = np.expand_dims(emo_img, 0)
            out = emotion([emo_img])[emotion_out]
            label = labels[np.argmax(out)]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return image

gr.Interface(
    fn=detect_emotion,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs="image"
).launch(share=True)
