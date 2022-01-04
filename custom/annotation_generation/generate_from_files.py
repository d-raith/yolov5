from typing import List

from PIL import Image
from torch.backends import cudnn
from tqdm import tqdm

from custom.yolo_detector import YoloParams, YoloDetector, Yolo5Result

import pandas as pd
import numpy as np

from remote_api.folder import Folder
from remote_api.loaders import VideoReader


def detect(files, yolo_params: YoloParams):
    detector = YoloDetector(yolo_params)

    cudnn.benchmark = True  # set True to speed up constant image size inference

    results = []

    for file in tqdm(files, desc="Detecting..."):
        image = np.asarray(Image.open(file))

        result = detector.yolo.get_detections([image])[0]

        result.img_path = file
        results.append(result)

    return results


def save_to_csv(results: List[Yolo5Result]):
    pass


def create_annotations(file_paths, output_dir: Folder, yolo_params: YoloParams):
    detections = detect(file_paths, yolo_params)

    for idx, result in tqdm(enumerate(detections)):
        Image.fromarray(result.img_source).save(output_dir.get_file_path(f"{idx}.jpg"))
        result.get_label_df().to_csv(output_dir.get_file_path(f"{idx}.txt"), sep=" ", header=False, index=False)



def video_to_images(video_src, format="jpg", grayscale=False):
    reader = VideoReader(video_src, auto_grayscale=grayscale)
    out = Folder(video_src.split(".")[0] + "_images", create=True)

    for idx, frame in tqdm(enumerate(reader)):
        Image.fromarray(frame).save(out.get_file_path(f"{idx}.{format}"))
    return out.path()


if __name__ == '__main__':
    # print(video_to_images("E:/repos/deepsort_2/laa-video-data/211215_111038_test_3s.mov"))

    images_folder = Folder("E:/repos/deepsort_2/laa-video-data/211215_111038_test_3s_images")

    params = YoloParams(conf=0.65, augment=True, yolo_weights="configurations/laa/best.pt")

    annotation_folder = Folder("./output", create=True)
    print(annotation_folder)

    create_annotations(images_folder.get_files(filter_extensions=['jpg']), annotation_folder, yolo_params=params)
