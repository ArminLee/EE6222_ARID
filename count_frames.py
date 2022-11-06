import os
import cv2

def count_frames():
    video_path = '/home/ymli/Programs/Course/6222/Baseline/dataset/ARID_dark/raw/train/Walk/'
    videos = os.listdir(video_path)  # 返回指定路径下的文件和文件夹列表。
    for video_name in videos:  # 依次读取视频文件
        vc = cv2.VideoCapture(video_path + video_name)
        frame_num = vc.get(7)
        print(frame_num)
        vc.release()

count_frames()