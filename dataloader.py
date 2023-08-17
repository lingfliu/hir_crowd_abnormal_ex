import os
import glob
import numpy as np
import scipy.io as sio
import skvideo
import cv2
import time
from multiprocessing import Pool

label_collapsed_mat = 'label_collapsed.mat'
video_root = '../../data/Motion_Emotion_Dataset'  # Path to the directory containing the videos
video_resize = 0.25  # Resize the video to this size


def label_load(label_mat='label_collapsed.mat'):
    mat = sio.loadmat(label_mat)
    return mat['label_collapsed']


def flow_calc(video_file, flow_path):
    # Get the video name
    video_name = os.path.splitext(os.path.basename(video_file))[0]

    # Get the flow file name
    flow_filename = os.path.join(flow_path, video_name)

    tic = time.time()
    # Load the video
    cam = cv2.VideoCapture(video_file)

    ret, prev = cam.read()
    if not ret:
        print('Failed to load video: ' + video_file)
        return -1
    # resize for speed-up
    prev = cv2.resize(prev, (0, 0), fx=video_resize, fy=video_resize)
    prevgrey = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prevflow = np.zeros((prev.shape[0], prev.shape[1], 2))
    fps = cam.get(cv2.CAP_PROP_FPS)
    frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)

    flows = np.empty((0, prev.shape[0], prev.shape[1], 2))
    frame_idx = 0

    while cam.isOpened():
        ret, img = cam.read()
        if not ret:
            break
        img = cv2.resize(img, (0, 0), fx=video_resize, fy=video_resize)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgrey, grey, None, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        np.append(flows, np.array([flow]), axis=0)

        # update image buffer
        prev = img
        prevgrey = grey
        prevflow = flow

        toc = time.time()
        frame_idx += 1
        print('calculating flow of video file {0}, frame {1}, time cost: {2}'.format(video_file, frame_idx,
                                                                                     toc - tic))
        tic = toc

    # Save the flow to disk
    np.save(flow_filename, flows)
    return 0


def flow_precalc_seq(video_root, flow_root):
    """
    Pre-calculate the optiflows of dataset in single process way
    """
    if not os.path.exists(flow_root):
        os.makedirs(flow_root)

    video_files = glob.glob(os.path.join(video_root, '*.mp4'))

    for video_file in video_files:
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        flow_file = os.path.join(flow_root, video_name, '.npy')
        if not os.path.exists(flow_file):
            flow_calc(video_file, flow_root)
    return 0

def flow_load(flow_root):
    """load flow files """
    # Get a list of all the flow files
    flo_files = glob.glob(os.path.join(flow_root, '*.flo'))

    flow_all = np.empty(0, )
    flow_index = []
    fidx = 0
    for file in flo_files:
        flows = np.load(file)
        # add all flows
        np.append(flow_all, flows, axis=0)
        # mark flow frames
        # repeatly add the index of the video file
        idx_list = [fidx, ] * flows.shape[0]
        flow_index.append(zip(idx_list, range(flows.shape[0])))

    return flow_all, flow_index


if __name__ == '__main__':
    # label_collapsed = label_load()
    # labels = []
    # for label in label_collapsed:
    #     labels.append(label[2])

    flow_root = '../../data/Motion_Emotion_Dataset/flows'
    # flows, flow_index = flow_load(flow_root)

    """this snippet precalculate flow via multiprocessing methods"""
    # pool = Pool(8)
    # if not os.path.exists(flow_root):
    #     os.makedirs(flow_root)
    # video_files = glob.glob(os.path.join(video_root, '*.mp4'))
    # for video_file in video_files:
    #     pool.apply_async(flow_calc, args=(video_file, flow_root,))
    #
    # while True:
    #     time.sleep(1)

    flow_precalc_seq(video_root, flow_root)
