from typing import List

from PIL import Image
from torch.backends import cudnn
from tqdm import tqdm

from custom.yolo_detector import YoloParams, YoloDetector, Yolo5Result

import pandas as pd
import numpy as np

from remote_api.folder import Folder
from remote_api.loaders import VideoReader
from settings import REPO_ROOT


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def detect(files, yolo_params: YoloParams, batch_size=500):
    detector = YoloDetector(yolo_params)

    cudnn.benchmark = True  # set True to speed up constant image size inference

    batches = list(chunks(files, batch_size))

    for batch in batches:
        yield detector.detect_paths(batch)


def create_annotations(file_paths, output_dir: Folder, yolo_params: YoloParams, batch_size=750, write_images=True):
    file_paths = sorted(file_paths, key=lambda f: int(f.split(os.path.sep)[-1].split(".")[0]))
    frame_id = 0

    for batch_result in tqdm(
            detect(file_paths, yolo_params, batch_size=batch_size),
            desc="Process batch",
            total=int(len(file_paths) / batch_size) + 1, position=1):

        for result in tqdm(batch_result, desc="Write batch", total=len(batch_result), position=0):
            if write_images:
                Image.fromarray(result.img_source).save(output_dir.get_file_path(f"{frame_id}.jpg"))
            result.get_label_df(normalize=True).to_csv(output_dir.get_file_path(f"{frame_id}.txt"), sep=" ",
                                                       header=False,
                                                       index=False)

            frame_id += 1


def video_to_images(video_src, format="jpg", grayscale=False, out_folder=None):
    reader = VideoReader(video_src, auto_grayscale=grayscale)
    out = out_folder or Folder(video_src.split(".")[0] + "_images", create=True)

    for idx, frame in tqdm(enumerate(reader), total=reader.efc()):
        Image.fromarray(frame).save(out.get_file_path(f"{idx}.{format}"))
    return out


if __name__ == '__main__':
    vid_folder = REPO_ROOT.get_parent().make_sub_folder("laa-video-data")
    vid_file = vid_folder.get_file_path("211215_110858_test.avi")
    import os

    if not os.path.exists(vid_file):
        raise ValueError("File not found " + vid_file)

    images_folder = vid_folder.make_sub_folder(vid_file.split(os.path.sep)[-1].split(".")[0] + "_frames")

    images_folder = video_to_images(vid_file, out_folder=images_folder)

    params = YoloParams(conf=0.8, augment=False, yolo_weights="configurations/laa_high_fps/best.pt")

    files = images_folder.get_files(filter_extensions=['jpg'])

    create_annotations(files, images_folder, yolo_params=params, write_images=False)
