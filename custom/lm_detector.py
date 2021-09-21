import argparse
import os

import numpy as np
import torch
from PIL import Image
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, \
    scale_coords, set_logging
from utils.plots import Annotator

from utils.torch_utils import select_device, load_classifier, time_sync

print(np.__version__)
weight_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best-yolo-v1-3k.pt")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=544, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    arg = ['--weights', weight_path, '--conf-thres', '0.45']

    opt = parser.parse_args(arg)

    check_requirements(exclude=('pycocotools', 'thop'))
    return opt


class LmDetector():
    def __init__(self):
        self.args = get_args()
        self._load()

    def _load(self):
        opt = self.args
        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        # Initialize
        set_logging()
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(
                self.device).eval()

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def _run_detection(self, imgs, conf_th=0.45, iou_thres=0.45, bbox_color=(0, 230, 255)):

        self.colors[0] = bbox_color
        results = []
        for img in imgs:

            img0 = img.copy()

            img = letterbox(img, (self.args.img_size, self.args.img_size), stride=32)[0]
            if img.shape[2] <= 10:
                img = img.transpose((2, 0, 1))

                # Padded resize

            # Convert
            # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_sync()
            pred = self.model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_th, iou_thres, classes=self.args.classes,
                                       agnostic=self.args.agnostic_nms, max_det=3000)
            t2 = time_sync()

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, imgs)

            annotator = Annotator(img0, line_width=1, pil=True)
            # Process detections
            for i, det in enumerate(pred):  # detections per image

                gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                s = '%gx%g ' % img.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        print(self.colors[int(cls)])
                        annotator.box_label(xyxy, label, color=tuple(self.colors[int(cls)]))
                        # plot_one_box(xyxy, im=img0, label=label if cls != 0 else None, color=self.colors[int(cls)], line_thickness=2)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                img0 = annotator.result()
                results.append((img0.squeeze(), det.cpu().numpy()))

            return results

    def detect(self, imgs, conf_th=None, iou_thres=None):
        conf_th = conf_th or 0.45
        iou_thres = iou_thres or 0.45

        return self._run_detection(imgs, conf_th=conf_th, iou_thres=iou_thres)


if __name__ == '__main__':
    test = LmDetector()

    import numpy as np

    img = np.asarray(Image.open("/Users/Dennis/Desktop/rpistage_test_data_210420/40x/test/61.8=lp_0.2921875=z.png"))

    result = test.detect([img])
    import matplotlib.pyplot as plt

    plt.switch_backend("tkagg")

    for r_img, r_det in result:
        plt.imshow(r_img)
        plt.show()

    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    #
    # arg = ['--weights', 'runs/train/exp24/weights/best.pt', '--conf-thres', '0.4', '--img-size', '544', '--source',
    #        '/Users/Dennis/Desktop/rpistage_test_data_210420/40x/test']
    #
    # opt = parser.parse_args(arg)
    #
    # check_requirements(exclude=('pycocotools', 'thop'))
    #
    # with torch.no_grad():
    #     if opt.update:  # update all models (to fix SourceChangeWarning)
    #         for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
    #             detect()
    #             strip_optimizer(opt.weights)
    #     else:
    #         detect()
