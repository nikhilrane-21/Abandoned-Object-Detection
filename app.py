import cv2
import streamlit as st
import tempfile
import base64
import numpy as np
from collections import Counter, defaultdict


st.set_page_config(page_title="Abandoned Object Detection", page_icon="🤖")
hide_streamlit_style = """
            <style>
            .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
            .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
            .viewerBadge_text__1JaDK {display: none;}
            MainMenu {visibility: hidden;}
            header { visibility: hidden; }
            footer {visibility: hidden;}
            #GithubIcon {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Abandoned Object Detection")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mkv", "mov"])
threshold1 = st.sidebar.slider('Min Threshold', min_value=0, max_value=255, value=10)
threshold2 = st.sidebar.slider('Max Threshold', min_value=0, max_value=255, value=200)

if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    cap = cv2.VideoCapture(temp_file.name)
    _, firstframe = cap.read()
    consecutiveframe = 20
    track_temp = []
    track_master = []
    track_temp2 = []
    top_contour_dict = defaultdict(int)
    obj_detected_dict = defaultdict(int)
    frameno = 0

    stframe = st.empty()
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret==0:
            break
        frameno = frameno + 1
        frame_diff = cv2.absdiff(firstframe, frame)
        edged = cv2.Canny(frame_diff, threshold1, threshold2)
        kernel2 = np.ones((5, 5), np.uint8)
        thresh2 = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel2, iterations=2)
        (cnts, _) = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mycnts = []
        for c in cnts:
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            if M['m00'] == 0:
                pass
            else:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if cv2.contourArea(c) < 200 or cv2.contourArea(c) > 20000:
                    pass
                else:
                    mycnts.append(c)
                    (x, y, w, h) = cv2.boundingRect(c)
                    sumcxcy = cx + cy
                    track_temp.append([cx + cy, frameno])
                    track_master.append([cx + cy, frameno])
                    countuniqueframe = set(j for i, j in track_master)
                    if len(countuniqueframe) > consecutiveframe or False:
                        minframeno = min(j for i, j in track_master)
                        for i, j in track_master:
                            if j != minframeno:
                                track_temp2.append([i, j])
                        track_master = list(track_temp2)
                        track_temp2 = []
                    countcxcy = Counter(i for i, j in track_master)
                    for i, j in countcxcy.items():
                        if j >= consecutiveframe:
                            top_contour_dict[i] += 1
                    if sumcxcy in top_contour_dict:
                        if top_contour_dict[sumcxcy] > 100:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                            cv2.putText(frame, '%s' % ('Threat Object'), (x + w + 20, y + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(frame, 'Area:' + str(area), (x + w + 20, y + 45),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            obj_detected_dict[sumcxcy] = frameno
        for i, j in list(obj_detected_dict.items()):
            if frameno - obj_detected_dict[i] > 200:
                obj_detected_dict.pop(i)
                top_contour_dict[i] = 0
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode()

        stframe.markdown(f'<img src="data:image/jpeg;base64,{frame_base64}"/>', unsafe_allow_html=True)
    cap.release()
