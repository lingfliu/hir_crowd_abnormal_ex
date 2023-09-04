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


def label_load_trunc(label_mat='label_collapsed.mat'):
    mat = sio.loadmat(label_mat)
    labels = mat['label_collapsed']

    labels_trunc = np.empty((0, 3))
    file_idx = -1
    frame_idx = -1
    for l in labels:
        if file_idx == l[0]:
            if frame_idx > 0:
                labels_trunc = np.append(labels_trunc, np.array([l]), axis=0)
            frame_idx += 1
        else:
            frame_idx  = 0
            file_idx = l[0]
    return labels_trunc


def flow_calc(video_file, flow_path):
    # Get the video name
    video_name = os.path.splitext(os.path.basename(video_file))[0]

    # Get the flow file name
    flow_filename = os.path.join(flow_path, video_name)

    tic = time.time()
    # Load the video
    cam = cv2.VideoCapture(video_file)
    # cam = cv2.VideoCapture(0)

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
        flow_filename = os.path.join(flow_root, video_name)
        flow_file = flow_filename + '.npy'
        if not os.path.exists(flow_file):
            # flow_calc(video_file, flow_root)

            tic = time.time()
            cap = cv2.VideoCapture(video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            ret, img = cap.read()
            if not ret:
                print('Failed to load video: ' + video_file)
                continue

            # resize for speed-up
            img = cv2.resize(img, (0, 0), fx=video_resize, fy=video_resize)
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flow = np.zeros((img.shape[0], img.shape[1], 2))

            flows = np.empty((0, img.shape[0], img.shape[1], 2))

            prev_img = img
            prev_img_grey = img_grey # buffer previous frame
            prev_flow = flow

            frame_idx = 0
            while cap.isOpened():
                ret, img = cap.read()


                if not ret:
                    break
                img = cv2.resize(img, (0, 0), fx=video_resize, fy=video_resize)
                img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # visualize video
                # cv2.imshow('frame', img_grey)
                # if cv2.waitKey(2) & 0xFF == ord('q'):
                #     break

                flow = cv2.calcOpticalFlowFarneback(prev_img_grey, img_grey, None, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                # flows[frame_idx, :, :, :] = np.append(flows, np.array([flow]), axis=0)
                flows = np.append(flows, np.array([flow]), axis=0)

                prev_img = img
                prev_img_grey = img_grey
                prev_flow = flow

                toc = time.time()
                frame_idx += 1
                print('calculating flow of video file {0}, frame {1}, time cost: {2}'.format(video_file, frame_idx,
                                                                                             toc - tic))
                tic = toc

            np.save(flow_filename, flows)
    return 0

def flow_load(flow_root):
    """load flow files """
    # Get a list of all the flow files
    flow_files = glob.glob(os.path.join(flow_root, '*.npy'))

    frames_all = 0
    for file in flow_files:
        flow = np.load(file)
        frames_all += flow.shape[0]
        flow_all = np.empty((0, flow.shape[1], flow.shape[2], flow.shape[3]))
        info = 'calculating frames of flow file {0} = {1}'.format(file, flow.shape[0])
        print("\r" + info, end="")
    print('\n')

    flow_all = np.zeros((frames_all, flow.shape[1], flow.shape[2], flow.shape[3]))

    flow_index = []
    fidx = 0
    for file in flow_files:
        flows = np.load(file)
        # add all flows
        # flow_all = np.append(flow_all, flows, axis=0)
        flow_all[fidx:fidx + flows.shape[0], :, :, :] = flows
        # mark flow frames
        # repeatly add the index of the video file
        idx_list = [fidx, ] * flows.shape[0]
        [flow_index.append([fidx, i]) for i in range(flows.shape[0])]

        info = 'loading flow file {0}'.format(file)
        print("\r" + info, end="")

    return flow_all, flow_index


if __name__ == '__main__':
    label_collapsed = label_load()
    labels = []
    for label in label_collapsed:
        labels.append(label[2])

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
