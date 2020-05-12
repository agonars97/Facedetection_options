#!/usr/bin/python3
import os
import cv2
import sys
import queue
import argparse
import threading
import numpy as np
from time import time
from pathlib import Path
from datetime import datetime
from threading import Thread, Lock
import traceback
sys.path.append(str(Path(__file__).absolute().parent.parent.parent/'lib'))
print(sys.path)
from blueeyes.face_recognition import FaceDetector

shared_queue = queue.Queue()
lock = Lock()

parser = argparse.ArgumentParser('Dataset Builder Program')
parser.add_argument('filename', help='Filename for the output video folder.')
parser.add_argument('--user', help='Username for IP camera.')
parser.add_argument('--password', help='Password for IP camera.')
parser.add_argument('--ip', help='IP of the camera.')
parser.add_argument('--builtin-cam', action='store_true', help='Use internal camera.')
parser.add_argument('--output', help='Image output folder')
parser.add_argument('--skip', help='Number of frames to skip', default=0)
args = parser.parse_args()

OUTPUT_DIR = Path(args.output)
dt = datetime.now()
OUTPUT_DIR = OUTPUT_DIR / dt.strftime('%Y%m%d')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# if os.path.exists(args.filename):
#     print(f'{args.filename} exists. Choose a different filename or rename')
#     sys.exit(0)

if args.builtin_cam:
    print('0')
    cam_url = 0
else:
    cam_url = f'rtsp://{args.user}:{args.password}@{args.ip}:554/Streaming/Channels/101'

cap = cv2.VideoCapture(cam_url)
# detector.detect(np.zeros((2,2,3)))
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
FPS = 25
print(FRAME_HEIGHT,FRAME_WIDTH,FPS)

# static variables decoration
def static(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

# reading and showing image from camera thread
def get_frame(q):
    while cap.isOpened():
        ret, frame = cap.read()
        if ret==True:
            frame_to_show = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('frame', frame_to_show)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                while cv2.waitKey(1) & 0xFF != ord(' '):
                    pass
            if key == ord('q'):
                break
            q.put(frame)
            q.task_done()
        else:
            print("Get frame error!")
    # Release everything if job is finish
    cap.release()
    print('Get frame done.')
# @static(t=0)
def process_frame(q):
    detector = FaceDetector('mtcnn', min_face_size=20)
    t = 0
    start_time = None
    first_status = None
    first_time = None
    while cap.isOpened() or not q.empty():
        try:
            for _ in range(args.skip+1):
                if q.empty():
                    break
                frame = q.get(False)
            # out.write(frame)
            # lock.acquire()
            boxes = detector.detect(frame)
            if boxes and first_status == None:
                first_time = 'true'
            # split to new folder after no face detected in 5s
            if (time() - t > 5 and boxes) or first_time == True:
                dt = datetime.now()
                folder_name = dt.strftime('%H%M%S')
                subdir = OUTPUT_DIR / folder_name
                os.makedirs(subdir)
                first_time = False
            # lock.release()
            if start_time == None:
                video_name = dt.strftime('%H%M%S')
                out = cv2.VideoWriter(os.path.join(args.filename, f'{video_name}.avi'), cv2.VideoWriter_fourcc(*'MJPG'), FPS, (FRAME_WIDTH, FRAME_HEIGHT))
                start_time = time()              

            elif time() - start_time > 3600*2:
                video_name = dt.strftime('%H%M%S')
                out.release()
                out = cv2.VideoWriter(os.path.join(args.filename, f'{video_name}.avi'), cv2.VideoWriter_fourcc(*'MJPG'), FPS, (FRAME_WIDTH, FRAME_HEIGHT))  
                start_time = time()              
            if boxes:
                out.write(frame)
                for x1,y1,x2,y2 in boxes:
                    t = time()
                    face_im = frame[y1:y2, x1:x2, :]
                    cv2.imwrite(f'{subdir}/{id(face_im)}.jpg', face_im)
            
            
        except Exception as e:
            traceback.print_exc()
            input()
        # detector.debug(frame)
    # out.release()
    print('Process frame done.')
    # Define the codec and create VideoWriter object
    

    


if __name__ == '__main__':
    thread1 = threading.Thread(target=get_frame, args=(shared_queue,))
    thread2 = threading.Thread(target=process_frame, args=(shared_queue,))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    shared_queue.join()