import re
from typing import List

from PIL import Image

from torch.backends import cudnn
from tqdm import tqdm

from custom.yolo_detector import YoloParams, YoloDetector, Yolo5Result

import pandas as pd
import numpy as np

from remote_api.folder import Folder
from remote_api.video_utils import VideoReader


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


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


def create_annotations(file_paths, output_dir: Folder, yolo_params: YoloParams):
    file_paths = natural_sort(file_paths)
    detections = detect(file_paths, yolo_params)

    for idx, result in tqdm(enumerate(detections)):
        Image.fromarray(result.img_source).save(output_dir.get_file_path(f"{idx}.jpg"))
        result.get_label_df(normalize=True).to_csv(output_dir.get_file_path(f"{idx}.txt"), sep=" ", header=False,
                                                   index=False)


def label_sample_files(file_paths, output_dir: Folder, yolo_params: YoloParams, show_only=False):
    print(f"{len(file_paths)} images to process")
    file_paths = natural_sort(file_paths)
    detections_batch = list(detect(file_paths, yolo_params))
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    # plt.ion()
    # f, ax = plt.subplots(1, 1)
    results = []
    for b_id, b_results in tqdm(enumerate(detections_batch), leave=False):
        for r in b_results:
            results.append(r)

    for idx, result in tqdm(enumerate(results)):
        img = result.annotate()

        # if show_only:
        #     ax.cla()
        #     ax.imshow(img)
        #     plt.pause(0.05)
        #     continue
        Image.fromarray(img).save(output_dir.get_file_path(f"{idx}_ann.jpg"))

        if result.detections.shape[0] == 0:
            print("No detections on frame", idx)

        print("show")
        # result.get_label_df(normalize=True).to_csv(output_dir.get_file_path(f"{result.file_name}.txt"), sep=" ",
        #                                           header=False,
        #                                           index=False)


def video_to_images(video_src, frame_folder_out=None, format="jpg", grayscale=False, out_folder=None):
    reader = VideoReader(video_src, auto_grayscale=grayscale)
    out = frame_folder_out or Folder(video_src.split(".")[0] + "_images", create=True)

    for idx, frame in tqdm(enumerate(reader), total=reader.efc()):
        Image.fromarray(frame).save(out.get_file_path(f"{idx}.{format}"))
    return out


if __name__ == '__main__':
    src_file = "/home/raithd/data/streamlit_data/220530_151825_test_control/220530_151825_test_control.mp4"
    image_out_folder = Folder(src_file.split(".")[0], create=True)
    annotation_folder = Folder("./output/samples", create=True)

    # video_to_images(src_file, frame_folder_out=image_out_folder)

    params = YoloParams(conf=0.5, augment=True, iou=0.3, yolo_weights="configurations/laa_960540_ofm_v1/best.pt")
    print(params)
    print(annotation_folder)

    label_sample_files(image_out_folder.get_files(filter_extensions=['jpg']), annotation_folder, yolo_params=params)

    # create_annotations(image_out_folder.get_files(filter_extensions=['jpg']), annotation_folder, yolo_params=params)
