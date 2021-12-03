import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from random import shuffle
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from numpy import random
from tqdm import tqdm

from custom.filesystem_utils import get_batch_iterator, split_filename_extension
from models.experimental import attempt_load
from remote_api.folder import Folder
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, set_logging, xyxy2xywh, xywh2xyxy
from utils.plots import Annotator
from utils.torch_utils import select_device, load_classifier, time_sync

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class YoloParams:
    yolo_weights: str = "best-yolo-v1-3k.pt"
    augment: bool = False
    classes: int = None
    agnostic_nms: bool = False
    imgsz: int = 640
    iou: float = 0.25
    conf: float = 0.4
    max_detections = 30000
    classify = False

    def load_weights(self, device="cpu"):
        path = os.path.join(WEIGHT_DIR, self.yolo_weights)
        if not os.path.exists(path):
            raise ValueError(f"Unable to find file {path}")
        return attempt_load(weights=path, map_location=device)

    def properties(self):
        return self.__dict__.items()

    def to_df(self):
        return pd.DataFrame([self])

    @staticmethod
    def from_df(row):
        return YoloParams(**row.to_dict(orient="records")[0])


@dataclass
class Yolo5Result:
    img_path: str
    img_source: np.ndarray

    detections: np.ndarray
    params: YoloParams
    _annotated = False

    @property
    def file_name(self):
        return split_filename_extension(self.img_path)[0]

    @property
    def file_extension(self):
        return split_filename_extension(self.img_path)[1]

    def annotate(self, colors=[(200, 0, 0), ], names=['cell'], line_width=2, font_size=40, font='Arial.ttf', show_class=False,
                 conf_based_color=True):
        ann = Annotator(self.img_source, line_width, font_size, font)

        def get_color(cnf):
            if conf_based_color:
                if cnf - self.params.conf < 0.1:
                    return 200, 0, 0
                if cnf - self.params.conf < 0.25:
                    return 120, 120, 0
                return 0, 200, 0

        for idx, row in self.get_label_df(normalize=False, include_conf=True).iterrows():
            xyxy = xywh2xyxy(torch.tensor(row[['x', 'y', 'w', 'h']].values).view(1, 4)).squeeze().tolist()
            conf, cls = row.conf, row.cls

            name, color = names[int(cls) - 1], get_color(conf)
            if show_class:
                label = f'{name} {conf:.2f}'
            else:
                label = f'{conf:.2f}'

            ann.box_label(xyxy, label, color=tuple(color))

        return ann.result()

    def get_label_df(self, normalize=True, ignore_timestamp_area=True, include_conf=False):

        data = []

        for *xyxy, conf, cls in reversed(self.detections):
            xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4))

            if normalize:
                gn = torch.tensor(self.img_source.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                xywh = (xywh / gn).view(-1)  # normalized xywh

            x, y, w, h = xywh.squeeze().tolist()

            img_h, img_w = self.img_source.shape[:2] if not normalize else (1, 1)
            if ignore_timestamp_area and x <= 0.3 * img_w and y <= 0.1 * img_h:
                continue

            data.append(dict(cls=int(cls), x=x, y=y, w=w, h=h, conf=conf))
        data = pd.DataFrame(data)
        if include_conf:
            return data

        else:
            return data.drop("conf", axis=1)

    def plot_result(self, ax=None):
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        ax.imshow(self.annotate())
        ax.axis("off")

    def side_by_side(self, axis):
        axis[0].imshow(self.img_source), axis[1].imshow(self.annotate())
        axis[0].axis("off")
        axis[1].axis("off")
        axis[0].autoscale(enable=True, tight=True)
        axis[1].autoscale(enable=True, tight=True)


class Yolo5:
    def __init__(self, params: YoloParams):
        self.params = params
        self._load()

    def _load(self):
        # Initialize
        set_logging()
        self.device = select_device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = self.params.load_weights(self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.params.imgsz, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        if self.params.classify:
            # Second-stage classifier
            self.classifier = load_classifier(name='resnet101', n=2)  # initialize
            self.classifier.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(
                self.device).eval()

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # bg, cell, bud
        self.colors = [[0, 0, 0], [255, 0, 0], [0, 255, 0]] + [[random.randint(0, 255) for _ in range(3)] for _ in
                                                               self.names]

    def _inference_step(self, image_tensor):
        # Inference
        t1 = time_sync()
        pred = self.model(image_tensor, augment=self.params.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.params.conf, self.params.iou, classes=self.params.classes,
                                   agnostic=self.params.agnostic_nms, max_det=self.params.max_detections)
        t2 = time_sync()

        # Apply Classifier
        if self.params.classify:
            pred = apply_classifier(pred, self.classifier, image_tensor, image_tensor)

        return pred

    def _process_input_image(self, img, apply_letterbox=True):
        if apply_letterbox:
            img = letterbox(img, (self.params.imgsz, self.params.imgsz), stride=self.stride)[0]
        if img.shape[2] <= 10:
            img = img.transpose((2, 0, 1))
        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def get_detections(self, images, paths=None, pre_process=True) -> List[Yolo5Result]:

        paths = paths or [None] * len(images)

        # Padded resize

        results = []

        imgs = [self._process_input_image(image, apply_letterbox=pre_process).squeeze() for image in images]

        imgs = torch.stack(imgs)

        predictions = self._inference_step(imgs)

        for i, det in enumerate(predictions):
            # gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size

                det[:, :4] = scale_coords(imgs.shape[2:], det[:, :4], images[i].shape).round()

            result = Yolo5Result(paths[i], images[i], det.cpu().numpy(), self.params)
            results.append(result)

        return results


class YoloDetector:
    def __init__(self, params: YoloParams):
        self.yolo = Yolo5(params)

    def detect_images(self, *images, pre_process=True):
        results = self.yolo.get_detections(list(images), pre_process=pre_process)
        if len(images) == 1:
            return results[0]
        return results

    def detect_paths(self, img_paths):
        imgs = []
        for idx, path in enumerate(img_paths):
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Failed to load image {path}")
            imgs.append(img)

        img_results = self.yolo.get_detections(imgs, paths=img_paths)
        if len(imgs) == 1:
            return img_results[0]
        return img_results


def generate_labels(yolo: YoloDetector, paths, output_dir=f"{THIS_DIR}/label_output", cls_names=['cell']):
    out = Folder(output_dir)
    assert out.exists(), f"Path {output_dir} does not exist"
    day_id = datetime.now().strftime('%Y%m%d')
    out = out.make_sub_folder(day_id)

    num_train = int(np.floor(len(paths) * 0.7))

    shuffle(paths)

    train = paths[:num_train]
    valid = paths[num_train:]

    print(f"Processing {len(paths)} files ({len(train)} train, {len(valid)} valid)")

    out_train = out.make_sub_folder("train")
    out_val = out.make_sub_folder("valid")
    data_yml = "train: ../train/images" + "\n" \
               + "val: ../valid/images" + "\n\n" \
               + f"nc: {len(cls_names)}" + "\n" \
               + f"names: {str(cls_names)}\n"

    with open(out.get_file_path("data.yml"), "w") as f:
        f.write(data_yml)

    write = 0

    batches_train = list(get_batch_iterator(train, 10))

    for batch in tqdm(batches_train, desc="Processing training set"):
        for res in yolo.detect_paths(batch):
            fname = res.file_name

            img_path = out_train.get_file_path(fname + res.file_extension)
            csv_path = out_train.get_file_path(fname + ".txt")
            shutil.copy(res.img_path, img_path)
            res.get_label_df().to_csv(csv_path, sep=" ", header=False, index=False)
            print(write, img_path)
            write += 1

    batches_valid = list(get_batch_iterator(valid, 10))
    for batch in tqdm(batches_valid, desc="Processing validation set"):
        for res in yolo.detect_paths(batch):
            fname = res.file_name

            img_path = out_val.get_file_path(fname + res.file_extension)
            csv_path = out_val.get_file_path(fname + ".txt")
            shutil.copy(res.img_path, img_path)
            res.get_label_df().to_csv(csv_path, sep=" ", header=False, index=False)


if __name__ == '__main__':
    pass
    # import matplotlib
    #
    # matplotlib.use("TkAgg")
    #
    # iou = [0.2, 0.4, 0.6, 0.8]
    # conf = [0.4, 0.5, 0.6]
    # augment = [True, False]
    #
    # imgsz = [320, 480, 544, 640, 800]
    # data = list(itertools.product(iou, conf, augment, imgsz))
    # print(data)
    # params = [YoloParams(iou=d[0], conf=d[1], augment=d[2], imgsz=d[3]) for d in data]
    # print(params)
    # evaluations = []
    #
    # detector = YoloDetector(YoloParams())
    # num_frames = 20
    # image_paths = [f"E:/hartmann_test_videos/1/frames/20211126_135600{i:02d}.jpg" for i in range(num_frames)]
    #
    # for param in tqdm(params):
    #     detector.yolo.params = param
    #     results, annotations = detector.detect(image_paths)
    #
    #     detections = list(map(lambda r: len(r[1]), results))
    #     mean_det = (sum(detections) / len(detections)) if len(detections) > 0 else 0
    #     evaluations.append(
    #         {**param.to_df().to_dict(orient="records")[0], 'det': mean_det, 'anns': annotations, 'r': results})
    #
    # df = pd.DataFrame(evaluations)
    #
    # data_df = df.drop(['r', 'anns', 'yolo_weights'], axis=1)
    #
    # print(data_df.sort_values("det", ascending=False).head(50))
    #
    # # img = df[df.iou == 0.2].anns.item()[0].result()
    #
    # # plt.imshow(img)
    # # plt.show()
