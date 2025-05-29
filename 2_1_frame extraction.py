from glob import glob
import os
import numpy as np
import cv2
import operator

input_dir = 'UCF101_train'  # directory with class folders containing .avi videos
output_dir = 'UCF101_frames'
target_classes = [
    'ApplyEyeMakeup', 'Archery', 'BabyCrawling', 'Basketball', 'BenchPress',
    'Biking', 'Bowling', 'BoxingPunchingBag', 'CliffDiving', 'Diving',
    'Drumming', 'GolfSwing', 'Haircut', 'HorseRiding', 'JumpRope'
]
os.makedirs(output_dir, exist_ok=True)

for class_name in target_classes:
    class_path = os.path.join(input_dir, class_name)
    video_paths = glob(os.path.join(class_path, '*.avi'))

    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_folder = os.path.join(output_dir, class_name, video_name)
        os.makedirs(out_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(out_folder, f"{frame_idx:04d}.png")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1
        cap.release()