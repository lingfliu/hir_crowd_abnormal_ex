import torch
import numpy as np
import cv2

from array_tool import to_categorical
from model import Unet
from dataloader import label_load_trunc, flow_load
import random


label_collapsed = label_load_trunc() # 减掉第一帧的标签
labels = []
for label in label_collapsed:
    labels.append(label[2])
flow_root = '../../data/Motion_Emotion_Dataset/flows'
flows, flow_index = flow_load(flow_root)


idx = 0
for flow in flows:
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mag = mag / np.max(mag)
    heatmap = np.uint8(mag * 255)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    cv2.putText(heatmap, str(labels[idx]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('heatmap', heatmap)
    idx += 1

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
