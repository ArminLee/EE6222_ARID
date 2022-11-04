import os
import cv2

def save_video(file_dir, video_name):
    list=[]
    list = os.listdir(file_dir)
    # list.remove('Drink_3_13_')
    list.sort(key=lambda x:int(x.split('.')[0][12:]))
    img_dir = file_dir+'/'
    print(list)

    category_path = '/home/ymli/Programs/Course/6222/Baseline/dataset/ARID_Light/train/'
    video_dir = os.path.join(category_path, video_name)
    #os.makedirs(video_dir, exist_ok=True)
    video_path = category_path + video_name + '.mp4'
    print(video_path)
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30,(320,240))
    for i in range(1,len(list)):
        img=cv2.imread(img_dir+list[i-1])
        video.write(img)

    video.release()
    print('save success!')
