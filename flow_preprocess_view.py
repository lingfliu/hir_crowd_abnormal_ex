from dataloader import label_load
import os
import glob
import cv2
import time
import numpy as np
from dataloader import label_load, label_load_trunc, flow_load

"""
查看预处理后的光流
"""
flow_root = '../../data/Motion_Emotion_Dataset/flows'

labels = label_load_trunc()

# flows, flow_index = flow_load(flow_root)
# for flow in flows:
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
#     mag = mag / np.max(mag)
#     heatmap = np.uint8(mag * 255)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
#     cv2.imshow('heatmap', heatmap)
#
#     if cv2.waitKey(2) & 0xFF == ord('q'):
#         break

flow_files = glob.glob(os.path.join(flow_root, '*.npy'))
for file in flow_files:
    flows = np.load(file)
    idx = 0
    for flow in flows:
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mag = mag / np.max(mag)
        heatmap = np.uint8(mag * 255)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        cv2.putText(heatmap, str(file)+str(idx), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('heatmap', heatmap)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

        idx += 1

