from glob import glob
import os
import numpy as np
import cv2
import operator

data_path = 'UCF101_frames'
target_classes = [
    'ApplyEyeMakeup', 'Archery', 'BabyCrawling', 'Basketball', 'BenchPress',
    'Biking', 'Bowling', 'BoxingPunchingBag', 'CliffDiving', 'Diving',
    'Drumming', 'GolfSwing', 'Haircut', 'HorseRiding', 'JumpRope'
]

ORIGINAL_NUM = 276918
ORIGINAL_CLASSES = 101
NEW_CLASSES = 15

NUM = int((ORIGINAL_NUM / ORIGINAL_CLASSES) * NEW_CLASSES)  # ≈ 41154


from glob import glob
import os
import numpy as np
import cv2
import operator

data_path = 'UCF101_frames'
target_classes = [
    'ApplyEyeMakeup', 'Archery', 'BabyCrawling', 'Basketball', 'BenchPress',
    'Biking', 'Bowling', 'BoxingPunchingBag', 'CliffDiving', 'Diving',
    'Drumming', 'GolfSwing', 'Haircut', 'HorseRiding', 'JumpRope'
]

ORIGINAL_NUM = 276918
ORIGINAL_CLASSES = 101
NEW_CLASSES = 15

NUM = int((ORIGINAL_NUM / ORIGINAL_CLASSES) * NEW_CLASSES)  # ≈ 41154


video_folders = []
class_folders = glob(os.path.join(data_path, '*'))  # classes

for cls_folder in class_folders:
    vids = glob(os.path.join(cls_folder, '*'))  # videos inside class
    video_folders.extend(vids)

print(f"Total videos: {len(video_folders)}")

# Now video_folders[i] points to a folder of frames for one video
# Process frames inside each video folder
total_counts = 0
class_counts = []

for video_folder in video_folders:
    png_files = glob(os.path.join(video_folder, '*.png'))
    #print(f'{video_folder}: {len(png_files)} frames')
    class_counts.append(len(png_files))
    total_counts += len(png_files)
classes = glob(os.path.join(data_path, '**'))
classes = [cls for cls in classes if os.path.basename(cls) in target_classes]
#print(len(classes))

# count suppose frame counts of each class
"""total_counts = 0
class_counts = []
for i in range(len(classes)):
    png_files = glob(os.path.join(classes[i], '*.png'))
    print('class ' + str(i) + ': ' + str(len(png_files)))
    class_counts.append(len(png_files))
    total_counts += len(png_files)
for cls in classes:
    print(f"{cls}: {len(glob(os.path.join(cls, '*.png')))}")
"""
filtered_counts = []
for i in range(len(classes)):
    filtered_counts.append(int(float(NUM)*class_counts[i]/float(total_counts)))
    print('filtered class ' + str(i) + ': ' + str(filtered_counts[i]))

def psnr(x1, x2):
    MSE = np.mean(np.square(x1-x2))
    MSE = np.maximum(MSE, 1e-10)
    return 10 * np.log10(1 / MSE)



"""for training set"""
# calculate PSNR and sort
f0 = open('frame1.txt', 'w')
f1 = open('frame2.txt', 'w')
f2 = open('frame3.txt', 'w')
for i in range(len(classes)):
    print('filtering... ' + str(i))
    triplets_dict = []
    png_files = glob(os.path.join(classes[i], '*.png'))
    png_files = sorted(png_files)
    for j in range(1, len(png_files)-1):
        idx = int(png_files[j][-8:-4])
        if png_files[j-1] == (png_files[j][:-8] + str(idx-1).zfill(4) + '.png') \
            and png_files[j+1] == (png_files[j][:-8] + str(idx+1).zfill(4) + '.png'):
            img0 = cv2.imread(png_files[j-1]).astype(np.float32) / 255.0
            img1 = cv2.imread(png_files[j]).astype(np.float32) / 255.0
            img2 = cv2.imread(png_files[j+1]).astype(np.float32) / 255.0

            psnr0 = psnr(img0, img1)
            psnr1 = psnr(img1, img2)

            #triplets_dict.append((png_files[j], (psnr0 + psnr1) / 2.0))
            triplets_dict[png_files[i]] = (psnr0 + psnr1) / 2.0

    triplets_dict = sorted(triplets_dict, key=lambda tup: tup[1])

    usable_count = min(filtered_counts[i], len(triplets_dict))
    if usable_count == 0:
        print(f"Skipping video {i}: no valid frame triplets")
        continue

    print(f'class {i}, psnr threshold = {triplets_dict[i][j][1]}')

    for j in range(usable_count):
        idx = int(triplets_dict[j][0][-8:-4])
        f0.write('./' + triplets_dict[j][0][:-8] + str(idx-1).zfill(4) + '.png' + '\n')
        f1.write('./' + triplets_dict[j][0][:-8] + str(idx).zfill(4) + '.png' + '\n')
        f2.write('./' + triplets_dict[j][0][:-8] + str(idx+1).zfill(4) + '.png' + '\n')

f0.close()
f1.close()
f2.close()

