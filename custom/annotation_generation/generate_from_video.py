import glob
import os.path
import re
import shutil
from datetime import datetime
from typing import List

import cv2
import yaml
from PIL import Image

from torch.backends import cudnn
from tqdm import tqdm

from custom.annotation_generation.generate_from_files import video_to_images
from custom.yolo_detector import YoloParams, YoloDetector, Yolo5Result

import pandas as pd
import numpy as np

from remote_api.folder import Folder
from remote_api.video_utils import VideoReader, get_video_reader, draw_image_with_boxes
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


def create_annotations(file_paths, output_dir: Folder, yolo_params: YoloParams, fname_prefix=""):
    file_paths = natural_sort(file_paths)
    detections = detect(file_paths, yolo_params)

    for idx, result in tqdm(enumerate(detections)):
        Image.fromarray(result.img_source).save(output_dir.get_file_path(f"{fname_prefix}{idx}.jpg"))
        result.get_label_df(normalize=True).to_csv(output_dir.get_file_path(f"{fname_prefix}{idx}.txt"), sep=" ",
                                                   header=False,
                                                   index=False)


def video_iter(video_file, num_frames_per_sec, start_offset, yolo_params: YoloParams):
    detector = YoloDetector(yolo_params)

    cudnn.benchmark = True

    reader = get_video_reader(video_file)

    fps = reader.estimated_fps()

    skip = int(fps / num_frames_per_sec)

    print("skip", skip, 'offset', start_offset)
    for f_id, frame in tqdm(enumerate(reader), leave=False):
        if f_id < start_offset or f_id % skip != 0:
            continue
        result = detector.detect_images(frame)
        yield f_id, frame, result


def label_sample_files(video_file, num_frames_per_sec, min_obj_width, output_dir: Folder, yolo_params: YoloParams,
                       show_only=False, fname_prefix=""):
    if show_only:
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

        w, h = xywhs[:, 2], xywhs[:, 3]

        # plt.scatter(w, h)
        # plt.scatter(np.arange(len(h)), h)
        # plt.show()

        min_width = w > min_obj_width
        min_height = h > min_obj_width

        result.detections = result.detections[min_width | min_height]

        img = result.annotate()

        if show_only:
            ax.cla()
            ax.imshow(img)
            plt.pause(0.05)
        else:
            Image.fromarray(img).save(output_dir.get_file_path(f"{fname_prefix}{idx}_ann.jpg"))

            if result.detections.shape[0] == 0:
                print("No detections on frame", idx)

            result.get_label_df(normalize=True).to_csv(
                output_dir.get_file_path(f"{fname_prefix}{result.file_name}.txt"), sep=" ",
                header=False,
                index=False)


def save_image(image, target_path):
    Image.fromarray(image).save(target_path)


def create_annotation(result, path, min_object_size=None):
    output_df = result.get_label_df(normalize=True)
    if min_object_size:
        obj_size_norm_height, obj_size_norm_width = min_object_size / result.img_source.shape[:2]

        matches_height = output_df['h'] >= obj_size_norm_height
        matches_width = output_df['w'] >= obj_size_norm_width

        output_df = output_df[matches_width | matches_height]

    output_df.to_csv(path, sep=" ", header=False, index=False)
    return output_df


def create_annotated_image(image, output_df):
    im_height, im_width = image.shape[:2]
    xywhs = output_df[['x', 'y', 'w', 'h']].copy()
    xywhs.x *= im_width
    xywhs.y *= im_height
    xywhs.w *= im_width
    xywhs.h *= im_height

    im_out = image.copy()

    im_out = draw_image_with_boxes(im_out, xywhs, scale=1)
    return im_out


def create_annotations_for_file(video_file, num_frames_per_sec, start_offset, yolo_params: YoloParams, output_dir,
                                fname_prefix=""):
    for frame_id, image, yolo_result in video_iter(video_file, num_frames_per_sec, start_offset, yolo_params):
        img_file_name = f"{fname_prefix}{frame_id}.jpg"

        image_path = output_dir.get_file_path(img_file_name)
        save_image(image, image_path)

        ann_path = image_path.replace(".jpg", ".txt")
        ann_df = create_annotation(yolo_result, ann_path)

        ann_image = create_annotated_image(image, ann_df)
        save_image(ann_image, output_dir.make_sub_folder("debug").get_file(img_file_name.replace(".jpg", "_ann.jpg")))


def create_dataset_folder(src_folders: List[Folder], output_folder: Folder):
    ds_folder = output_folder.make_sub_folder("dataset")
    src_folders = [f for f in src_folders if f.name != "dataset"]

    train_folder = ds_folder.make_sub_folder("train")
    debug = ds_folder.make_sub_folder("debug")
    images_folder = train_folder.make_sub_folder("images")
    labels_folder = train_folder.make_sub_folder("labels")

    valid_folder = ds_folder.make_sub_folder("valid")
    _ = valid_folder.make_sub_folder("images")
    _ = valid_folder.make_sub_folder("labels")

    for folder in tqdm(src_folders, desc="Generating dataset"):
        src_debug_folder = folder.make_sub_folder("debug", create=False)

        for image_path in folder.make_file_provider("jpg", abs_path=False):
            # TODO: Verify abs or rel path
            ann_path = image_path.replace(".jpg", ".txt")
            prefix = folder.name.split("_")[0] + folder.name.split("_")[1]
            try:
                shutil.copy(folder.get_file(image_path), images_folder.get_file(prefix + "_" + image_path))
                shutil.copy(folder.get_file(ann_path), labels_folder.get_file(prefix + "_" + ann_path))

                if src_debug_folder.exists():
                    print("Copy debug folder", src_debug_folder.path_abs)
                    shutil.copy(src_debug_folder.get_file(image_path.replace(".jpg", "_ann.jpg")),
                                debug.get_file(prefix + "_" + image_path))

            except Exception as e:
                print(e)

    lines = [
        "train: ./train/images\n",
        "valid: ./valid/images\n",
        "nc: 1\n",
        "names: ['cell']\n"
    ]

    with open(ds_folder.get_file("data.yaml"), 'w') as out:
        out.writelines(lines)
        # yaml.dump(yaml_data, out, default_flow_style=None, sort_keys=False)


def collect_from_dataset(video_storage_folder: Folder, output_folder: Folder, regenerate_annotation=True,
                         src_filter=None, num_frames_per_second=0.1, start_offset=0):
    entries = video_storage_folder.get_folders(abs_path=False, as_folder_obj=True)

    if src_filter is not None:
        entries = list(filter(src_filter, entries))

    params = YoloParams(conf=0.5, augment=True, iou=0.35, yolo_weights="configurations/laa_960540_ofm_v1/best.pt")
    if regenerate_annotation:
        for idx, folder_name in enumerate(entries):
            print("Process ", idx + 1, "of", len(entries))
            folder = video_storage_folder.make_sub_folder(folder_name, create=False)

            fname = folder.name + ".mp4"
            src_path = folder.get_file(fname)

            output_dir = output_folder.make_sub_folder(folder_name)

            create_annotations_for_file(src_path, num_frames_per_second, start_offset, params, output_dir)

    folders = output_folder.get_folders(as_folder_obj=True)
    create_dataset_folder(src_folders=folders, output_folder=output_folder)


def collect_from_single_input():
    src_file = "/home/raithd/data/streamlit_data/220601_125348_test-duv_treated/220601_125348_test-duv_treated.avi"

    image_out_folder = Folder(src_file.split(".")[0], create=True)
    annotation_folder = Folder("./output/samples", create=True)

    # video_to_images(src_file, frame_folder_out=image_out_folder)

    params = YoloParams(conf=0.5, augment=True, iou=0.35, yolo_weights="configurations/laa_960540_ofm_v1/best.pt")
    print(params)
    print(annotation_folder)

    label_sample_files(src_file, num_frames_per_sec=1, min_obj_width=22, output_dir=annotation_folder,
                       yolo_params=params, show_only=True)

    create_annotations(image_out_folder.get_files(filter_extensions=['jpg']), annotation_folder, yolo_params=params)


if __name__ == '__main__':
    src_folder = "/home/raithd/data/streamlit_data/"
    src_folder = Folder(src_folder)

    dataset_folder = Folder("./output/datasets/220622_v1", create=True)


    def folder_filter(folder): return "220601_1235" in folder


    collect_from_dataset(src_folder, output_folder=dataset_folder, src_filter=folder_filter, num_frames_per_second=0.05,
                         start_offset=5 * 60,
                         regenerate_annotation=True)
