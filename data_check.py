from dataloader import label_load
import os
import glob
import cv2

vid_root = '../../data/Motion_Emotion_Dataset'  # Path to the directory containing the videos

vid_files = glob.glob(os.path.join(vid_root, '*.mp4'))

labels = label_load()

for vid_file in vid_files:
    video_name = os.path.splitext(os.path.basename(vid_file))[0]
    cap = cv2.VideoCapture(vid_file)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_count = 0
    for idx in labels:
        if idx[0] == int(video_name):
            frame_count += 1

    print('frames of vid {0}={1}, labels for frames = {2}'.format(video_name, int(frames), frame_count))

