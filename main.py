import cv2 as cv
import numpy as np
import streamlit as st
from openvino import Core

st.title("AI Teacher - Face Expression Detection")

start = st.button("Start Webcam")

if start:
    cr = Core()
    face = cr.compile_model("intel/face-detection-0200/FP32/face-detection-0200.xml", "CPU")
    head = cr.compile_model("intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml", "CPU")
    emotion = cr.compile_model("intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml", "CPU")

    face_in = face.input(0)
    head_in = head.input(0)
    emotion_in = emotion.input(0)

    face_out = face.output(0)
    head_out = head.output(0)
    emotion_out = emotion.output(0)

    cam = cv.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        face_img = cv.resize(frame, (face_in.shape[3], face_in.shape[2]))
        face_img = face_img.transpose((2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)

        ans = face([face_img])[face_out]

        for dect in ans[0][0]:
            con = dect[2]
            if con > 0.6:
                xmi = int(dect[3]*w)
                ymi = int(dect[4]*h)
                xma = int(dect[5]*w)
                yma = int(dect[6]*h)
                xmi = max(0, xmi)
                ymi = max(0, ymi)
                xma = min(w, xma)
                yma = min(h, yma)

                cv.rectangle(frame, (xmi, ymi), (xma, yma), (0, 255, 0), 2)
                crop = frame[ymi:yma, xmi:xma]
                if crop.size == 0:
                    continue

                hp_img = cv.resize(crop, (60, 60))
                hp_img = hp_img.transpose((2, 0, 1))
                hp_img = np.expand_dims(hp_img, axis=0)
                hp_res = head([hp_img])
                yaw, pitch, roll = [v[0][0] for v in hp_res.values()]
                cv.putText(frame, f"Yaw:{yaw:.1f} Pitch:{pitch:.1f} Roll:{roll:.1f}",
                           (xmi, ymi-40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                em_img = cv.resize(crop, (64, 64))
                em_img = em_img.transpose((2, 0, 1))
                em_img = np.expand_dims(em_img, axis=0)
                em_res = emotion([em_img])
                em_in = np.argmax(em_res[emotion_out])
                em_lab = ["neutral", "happy", "sad", "surprise", "anger"][em_in]
                cv.putText(frame, f"Emotion:{em_lab}",
                           (xmi, ymi-20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        FRAME_WINDOW.image(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    cam.release()
