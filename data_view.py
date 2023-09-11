from dataloader import label_load
import os
import glob
import cv2
import time
import numpy as np

vid_root = '../../data/Motion_Emotion_Dataset'  # Path to the directory containing the videos

vid_files = glob.glob(os.path.join(vid_root, '*.mp4'))

labels_all = label_load()

for vid_file in vid_files:
    vid_idx = int(os.path.splitext(os.path.basename(vid_file))[0])
    lidx_start = -1
    lidx_stop = -1
    for idx, label in enumerate(labels_all):
        if label[0] == vid_idx:
            if lidx_start < 0:
                lidx_start = idx
        else:
            if lidx_start >= 0:
                lidx_stop = idx
                break
    video_name = os.path.splitext(os.path.basename(vid_file))[0]
    cap = cv2.VideoCapture(vid_file)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    ret, img = cap.read()
    next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame', img)
    prevgrey = next

    label = labels_all[lidx_start:lidx_stop]

    idx = 0

    hsv = np.zeros_like(img)
    hsv[..., 1] = 255

    while cap.isOpened():
        ret, img = cap.read()

        '''
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgrey, grey, None, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2) # magnitude of the flow
        # hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('flow', rgb)
        cv2.imshow('vid', next)
        prevgrey = grey
        '''

        if not ret:
            break
        cv2.putText(img, str(label[idx][2]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', img)
        idx += 1
        print('showing {0}'.format(idx))
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break