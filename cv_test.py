# import cv2
#
# video_file = 'test.mp4'
#
# cap = cv2.VideoCapture(video_file)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(2) & 0xFF == ord('q'):
#         break

import numpy as np

a = np.ones(10,20,2)
b = np.empty((0,10,20,2))

np.append(b, np.array([a]), axis=0)