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

# import numpy as np
#
# a = np.ones(10,20,2)
# b = np.empty((0,10,20,2))
#
# np.append(b, np.array([a]), axis=0)

import cv2
import numpy as np

video_file = '011.mp4'

cap = cv2.VideoCapture(video_file)
ret, frame1 = cap.read()
# frame1 = cv2.resize(frame1, (0, 0), fx=0.25, fy=0.25)
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 0
# hsv[...,0] = 0

while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    # frame2 = cv2.resize(frame2, (0, 0), fx=0.25, fy=0.25)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    mag = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    mag = mag / np.max(mag)
    heatmap = np.uint8(mag*255)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    # hsv[...,0] = ang*180/np.pi/2
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

    rgb = np.uint8(frame2)

    # stack the heatmap on top of the rgb image

    stacked = heatmap

    cv2.imshow('flow',rgb)
    cv2.imshow('heatmap',stacked)
    # cv2.imshow('vid', next)
    prvs = next

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
