import cv2
import glob
import os
from dataloader import label_load
from model import *
import numpy as np

from model import *
from PIL import Image, ImageDraw, ImageFont

lab_str = ["无人", "逃散", "冲突", "聚集", "异物", "正常"]

video_resize = 0.25

vid_root = '../../data/Motion_Emotion_Dataset'  # Path to the directory containing the videos

vid_files = glob.glob(os.path.join(vid_root, '*.mp4'))

font = cv2.FONT_HERSHEY_SIMPLEX  # 字体设置
text_position = (0, 25)
font_scale = 1
font_color = (255, 255, 255)  # 白色

labels_all = label_load()  # 加载所有标签

model = Fcn()  # 定义模型
model = model.to(model.device)
model.summarize()

weights_path = "weights/model_12.pth"  # 权重
weights = torch.load(weights_path)
model.load_state_dict(weights)
model.eval()

for video_file in vid_files:

    vid_idx = int(os.path.splitext(os.path.basename(video_file))[0])
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
    label = labels_all[lidx_start:lidx_stop]  # 取标签

    video_name = os.path.splitext(os.path.basename(video_file))[0]  # 001取出来编号
    cap = cv2.VideoCapture(video_file)  # 读取视频
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, img = cap.read()  # 取出
    img_model = cv2.resize(img, (0, 0), fx=video_resize, fy=video_resize)  # 输入模型的第一帧图片，缩放
    img_grey = cv2.cvtColor(img_model, cv2.COLOR_BGR2GRAY)  # 二值
    prev_img_grey = img_grey  # 第一帧视频

    frame = 1  # 第二帧视频
    while True:

        ret, img = cap.read()  # 取出视频， img是原始视频

        if not ret:
            print("读取视频结束")
            break

        img_model = cv2.resize(img, (0, 0), fx=video_resize, fy=video_resize)  # scale image to 0.25*0.25
        img_grey = cv2.cvtColor(img_model, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_img_grey, img_grey, None, 0.5, 5, 15, 3, 5, 1.1,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flow = flow[np.newaxis, :]
        flow = flow.transpose(0, 3, 1, 2)
        flow = torch.from_numpy(flow).float().to(model.device)

        pre = model(flow)  # 预测输出
        pre = torch.argmax(pre, 1).cpu().numpy()[0]

        # visualization
        imgPil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(imgPil)
        font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")
        draw.text((15, 5), '识别结果：'+lab_str[pre], (255, 0, 255), font=font)  # 预测数据
        draw.text((15, 30), '参考：'+lab_str[label[frame][2]], (66, 245, 90), font=font)  # gt标签
        imgTexted = cv2.cvtColor(np.asarray(imgPil), cv2.COLOR_RGB2BGR)

        cv2.imshow(video_name, imgTexted)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame += 1  #
        prev_img_grey = img_grey
    cap.release()
    cv2.destroyAllWindows()
