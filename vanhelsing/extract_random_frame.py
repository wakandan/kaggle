import argparse
import random
import commons
import os
import cv2
from logger import *

parser = argparse.ArgumentParser()
parser.add_argument('--video', help='path to the video file', required=True)
parser.add_argument('-p', help='probablity of taking a frame', default=0.01, type=float)
parser.add_argument('--pattern', help='output pattern', default='{filename}_{count}.jpg')
parser.add_argument('--folder', help='target folder', default='output')
args = parser.parse_args()
video_dir = args.video
target_folder = args.folder
output_pattern = args.pattern
random_seed = 42
random.seed(random_seed)
frame_count = 0
if os.path.isfile(video_dir):
    logging.info(f"target path {video_dir} is a file")
    filenames = [video_dir]
else:
    logging.info(f'target path {video_dir} is a directory')
    filenames = [os.path.join(video_dir, fn) for fn in os.listdir(video_dir)]

if not os.path.exists(target_folder):
    os.mkdir(target_folder)

for filename in filenames:
    org_file = filename
    filename = os.path.basename(filename)
    logging.info(f'processing file {filename}')
    filename = os.path.splitext(filename)[0]
    for frame in commons.video_frames(org_file):
        if random.random() < args.p:
            frame_count += 1
            target_file = output_pattern.format(**{'filename': filename, 'count': frame_count})
            target_file = os.path.join(target_folder, target_file)
            cv2.imwrite(target_file, frame)
            logging.info(f'written {target_file}')
    logging.info(f'total frame extracted {frame_count} for file {org_file}')
