import re
from typing import List

from PIL import Image

from torch.backends import cudnn
from tqdm import tqdm

from custom.yolo_detector import YoloParams, YoloDetector, Yolo5Result

import pandas as pd
import numpy as np

from remote_api.folder import Folder
from remote_api.video_utils import VideoReader, get_video_reader
from utils.general import xyxy2xywh


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


def label_sample_files(video_file, num_frames_per_sec, min_obj_width, output_dir: Folder, yolo_params: YoloParams,
                       show_only=False):
    import matplotlib
    matplotlib.use("TkAgg")

    detector = YoloDetector(yolo_params)

    cudnn.benchmark = True

    reader = get_video_reader(video_file)

    fps = reader.estimated_fps()

    skip = int(fps / num_frames_per_sec)
    print("skip", skip)

    from matplotlib import pyplot as plt
    # plt.ion()
    f, ax = plt.subplots(1, 1)
    results = []

    for f_id, frame in tqdm(enumerate(reader), leave=False):
        if f_id % skip != 0:
            continue
        result = detector.detect_images(frame)
        results.append(result)

    for idx, result in tqdm(enumerate(results)):

        xywhs = xyxy2xywh(result.detections[:, 0:4])

        w, h = xywhs[:, 2],xywhs[:, 3]

        #plt.scatter(w, h)
        #plt.scatter(np.arange(len(h)), h)
        #plt.show()

        min_width = w > min_obj_width
        min_height = h > min_obj_width

        result.detections = result.detections[min_width | min_height]

        img = result.annotate()

        if show_only:
            ax.cla()
            ax.imshow(img)
            plt.pause(0.05)
        else:
            Image.fromarray(img).save(output_dir.get_file_path(f"{idx}_ann.jpg"))

            if result.detections.shape[0] == 0:
                print("No detections on frame", idx)

            result.get_label_df(normalize=True).to_csv(output_dir.get_file_path(f"{result.file_name}.txt"), sep=" ",
                                                       header=False,
                                                       index=False)


if __name__ == '__main__':
    src_file = "/home/raithd/data/streamlit_data/220601_125348_test-duv_treated/220601_125348_test-duv_treated.avi"
    image_out_folder = Folder(src_file.split(".")[0], create=True)
    annotation_folder = Folder("./output/samples", create=True)

    # video_to_images(src_file, frame_folder_out=image_out_folder)

    params = YoloParams(conf=0.5, augment=True, iou=0.3, yolo_weights="configurations/laa_960540_ofm_v1/best.pt")
    print(params)
    print(annotation_folder)

    label_sample_files(src_file, num_frames_per_sec=1, min_obj_width=22, output_dir=annotation_folder, yolo_params=params, show_only=True)

    # create_annotations(image_out_folder.get_files(filter_extensions=['jpg']), annotation_folder, yolo_params=params)
