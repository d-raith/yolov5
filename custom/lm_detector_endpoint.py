from PIL import Image
from flask import Flask, request
from flask_restful import Resource, Api

from custom.lm_detector import LmDetector

app = Flask(__name__)
api = Api(app)

import numpy as np

detector = LmDetector()


# matplotlib.use("MacOSX")
class UploadImage(Resource):
    def post(self):
        print(request)
        file = Image.open(request.files['image'])

        if not file:
            return

        image = np.asarray(file)

        iou = request.form.get("iou")
        conf = request.form.get("conf")

        if isinstance(conf, str):
            conf = float(conf)
            print("Custom conf:", conf)

        if isinstance(iou, str):
            iou = float(iou)
            print("Custom iou:", iou)

        result = detector.detect([image], conf_th=conf, iou_thres=iou)

        ann_img, detections = result[0]

        print(result)

        return {'detections': detections.tolist()}


class TestEndpoint(Resource):
    def get(self):
        return {}


api.add_resource(UploadImage, '/')
api.add_resource(TestEndpoint, '/test')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5555)
