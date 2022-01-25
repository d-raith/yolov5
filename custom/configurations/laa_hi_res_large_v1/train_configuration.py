#!/usr/bin/env python
# coding: utf-8

import torch
import os
import train as yolo_train

YML_FOLDER = os.getcwd()
MODEL_YML = YML_FOLDER + os.path.sep + "yolov5s.yaml"
HYP_YML = YML_FOLDER + os.path.sep + "hyp_initial.yaml"

DATASET_PATH = YML_FOLDER + os.path.sep + "dataset"
DATASET_YML = DATASET_PATH + os.path.sep + "data.yaml"


def train_config():
    print('Setup complete. Using torch %s %s' % (
        torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

    # define number of classes based on YAML
    import yaml

    with open(MODEL_YML, 'r') as stream:
        num_classes = str(yaml.safe_load(stream)['nc'])
        print(num_classes)
    # !pip install wandb
    print("YML_FOLDER", YML_FOLDER)
    print("MODEL_YML", MODEL_YML)
    print("HYP_YML", HYP_YML)
    print("DATASET_PATH", DATASET_PATH)
    print("DATASET_YML", DATASET_YML)

    imgsz = 640
    batch = 12
    epochs = 500

    args = {'imgsz': imgsz, 'batch': batch, 'epochs': epochs, 'data': DATASET_YML, 'cfg': MODEL_YML, 'hyp': HYP_YML,
            'cache': ''}
    arg_input = []
    for k, v in args.items():
        arg_input.append(f"--{k}")
        arg_input.append(str(v))

    opt = yolo_train.parse_opt(args=arg_input)
    yolo_train.main(opt)
    import os, sys, pathlib

    # sys.path.append(str(pathlib.Path("../../../.").resolve()))
    # print(sys.path)

    # In[ ]:

    from utils.plots import plot_results  # plot results.txt as results.png

    # we can also output some older school graphs if the tensor board isn't working for whatever reason...

    results_folder = os.getcwd() + "/runs/train"

    last_run = sorted(os.listdir(results_folder))[-1]
    print(last_run)
    # last_run = "exp13"
    run_dir = results_folder + "/" + last_run


if __name__ == '__main__':
    train_config()
