import argparse
import itertools
import os
from dataclasses import dataclass
from itertools import permutations

import cv2
import numpy as np
import torch
from PIL import Image
from numpy import random
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import letterbox, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, \
    scale_coords, set_logging, xyxy2xywh
from utils.plots import Annotator
from utils.torch_utils import select_device, load_classifier, time_sync
import pandas as pd

print(np.__version__)
weight_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best-yolo-v1-3k.pt")


@dataclass
class YoloParams:
    yolo_weights: str = "best-yolo-v1-3k.pt"
    augment: bool = False
    classes: int = 1
    agnostic_nms: bool = False
    imgsz: int = 640
    iou: float = 0.6
    conf: float = 0.4
    max_detections = 3000
    classify = False

    def properties(self):
        return self.__dict__.items()

    def to_df(self):
        return pd.DataFrame([self])

    @staticmethod
    def from_df(row):
        return YoloParams(**row.to_dict(orient="records")[0])


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
        self.model = attempt_load(self.params.yolo_weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.params.imgsz, s=self.stride)  # check img_size
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

    def _process_input_image(self, img):

        img = letterbox(img, (self.params.imgsz, self.params.imgsz), stride=32)[0]
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

    def get_detections(self, images):

        img0 = images.copy()

        # Padded resize

        imgs = [self._process_input_image(image).squeeze() for image in images]
        imgs = torch.stack(imgs)

        predictions = self._inference_step(imgs)

        dets = []
        for i, det in enumerate(predictions):
            # gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(imgs.shape[2:], det[:, :4], img0[i].shape).round()
                dets.append(det.cpu())

        return list(zip(img0, dets))

    def annotate(self, image, detections):
        annotator = Annotator(image, line_width=2, pil=False)
        for *xyxy, conf, cls in reversed(detections):
            label = f'{self.names[int(cls)]} {conf:.2f}'

            annotator.box_label(xyxy, label, color=tuple(self.colors[int(cls)]))
        return annotator

    def write_label_csv(self, im0_shape, det, file):
        gn = torch.tensor(im0_shape)[[1, 0, 1, 0]]  # normalization gain whwh
        with open(file, 'a') as f:
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                f.write(('%g ' * len(line)).rstrip() % line + '\n')


class YoloDetector():
    def __init__(self, params: YoloParams):
        self.yolo = Yolo5(params)

    def _run_detection(self, paths):
        imgs = []

        for idx, path in enumerate(paths):
            img = cv2.imread(path)

            imgs.append(img)
        detections = self.yolo.get_detections(imgs)
        return detections

    def detect(self, imgs):
        img_results = self._run_detection(imgs)
        annotations = [self.yolo.annotate(img, det) for img, det in img_results]
        return img_results, annotations


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("TkAgg")

    iou = [0.2, 0.4, 0.6, 0.8]
    conf = [0.4, 0.5, 0.6]
    augment = [True, False]

    imgsz = [320, 480, 544, 640, 800]
    data = list(itertools.product(iou, conf, augment, imgsz))
    print(data)
    params = [YoloParams(iou=d[0], conf=d[1], augment=d[2], imgsz=d[3]) for d in data]
    print(params)
    evaluations = []

    detector = YoloDetector(YoloParams())
    num_frames = 20
    image_paths = [f"E:/hartmann_test_videos/1/frames/20211126_135600{i:02d}.jpg" for i in range(num_frames)]

    for param in tqdm(params):
        detector.yolo.params = param
        results, annotations = detector.detect(image_paths)


        detections = list(map(lambda r: len(r[1]), results))
        mean_det = (sum(detections) / len(detections)) if len(detections) > 0 else 0
        evaluations.append(
            {**param.to_df().to_dict(orient="records")[0], 'det': mean_det, 'anns': annotations, 'r': results})

    df = pd.DataFrame(evaluations)

    data_df = df.drop(['r', 'anns', 'yolo_weights'], axis=1)

    print(data_df.sort_values("det", ascending=False).head(50))

    # img = df[df.iou == 0.2].anns.item()[0].result()

    # plt.imshow(img)
    # plt.show()
