import cv2
import glob
import os
import numpy as np

from model import *
import time

'''全局参数设置'''
video_resize = 0.25
model = Fcn()  # 定义模型
model = model.to(model.device)
model.summarize()
weights_path = "weights/model_12.pth"  # 权重
weights = torch.load(weights_path)
model.load_state_dict(weights)
model.eval()

def infer(video_file):
    cap = cv2.VideoCapture(video_file)  # 读取视频
    ret, img = cap.read()  # 取出
    if ret == False:
        print("读取视频失败")
        cap.release()
        cv2.destroyAllWindows()
        return []

    img_model = cv2.resize(img, (0, 0), fx=video_resize, fy=video_resize)  # 输入模型的第一帧图片，缩放
    img_grey = cv2.cvtColor(img_model, cv2.COLOR_BGR2GRAY)  # 二值
    prev_img_grey = img_grey  # 第一帧视频
    frame = 1  # 第二帧视频

    labels = []

    while True:
        ret, img = cap.read()  # 取出视频， img是原始视频

        if not ret:
            # print("读取视频结束")
            break

        img_model = cv2.resize(img, (0, 0), fx=video_resize, fy=video_resize)  # 输入视频
        img_grey = cv2.cvtColor(img_model, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_img_grey, img_grey, None, 0.5, 5, 15, 3, 5, 1.1,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        flow = flow[np.newaxis, :]
        flow = flow.transpose(0, 3, 1, 2)
        flow = torch.from_numpy(flow).float().to(model.device)


        predict = model(flow)  # 预测输出
        predict = torch.argmax(predict, 1).cpu().numpy()[0]

        labels.append(predict)

        frame += 1  #
        prev_img_grey = img_grey

    cap.release()
    cv2.destroyAllWindows()
    return labels


if __name__ == '__main__':
    vid_root = '../../data/Motion_Emotion_Dataset'  # Path to the directory containing the videos
    vid_files = glob.glob(os.path.join(vid_root, '*.mp4'))
    for vid_file in vid_files:
        tic = time.time()
        print('processing', vid_file)
        predict = infer(vid_file)
        toc = time.time()
        print('time cost', toc - tic)
        print(predict)