import io
import os
from datetime import datetime
from uuid import uuid4

from PIL import Image
from flask import Flask, request
from flask_restful import Resource, Api


from custom.lm_detector import LmDetector

app = Flask(__name__)
api = Api(app)

import numpy as np

USE_MINIO = os.getenv("USE_MINIO", False)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT") or "localhost:9000"

MINIO_USER = os.getenv("MINIO_USER") or "test"
MINIO_SECRET = os.getenv("MINIO_SECRET") or "testtest"
MINIO_SECURE = os.getenv("MINIO_SECURE") or False
MINIO_USE_OBJECT_LOCK = os.getenv("MINIO_SECURE", 0) == 1
MINIO_BASE_BUCKET = os.getenv("MINIO_BASE_BUCKET")
if USE_MINIO:
    from minio import Minio
    from minio.commonconfig import GOVERNANCE
    from minio.helpers import ObjectWriteResult
    from minio.retention import Retention
    minio_client = Minio(MINIO_ENDPOINT, MINIO_USER, MINIO_SECRET, secure=MINIO_SECURE)
    base_exists = minio_client.bucket_exists(MINIO_BASE_BUCKET)
    if not base_exists:
        minio_client.make_bucket(MINIO_BASE_BUCKET, object_lock=MINIO_USE_OBJECT_LOCK)
    # verify connection
    [print("found bucket:", b.name) for b in minio_client.list_buckets()]

MINIO_URL = "https://" if MINIO_SECURE else "http://" + MINIO_ENDPOINT


def split_filename_extension(file_path):
    name, ext = os.path.splitext(file_path)
    return os.path.basename(name), ext


detector = LmDetector()

print("Object locking enabled:", MINIO_USE_OBJECT_LOCK)

def get_image_stream(np_img, img_format):
    img = Image.fromarray(np_img).convert('RGB')
    out_img = io.BytesIO()

    img.save(out_img, format=img_format)
    byte_count = out_img.tell()
    out_img.seek(0)
    return out_img, byte_count


def get_object_id(img_id_fname, object_prefix, img_format):
    img_id, ext = split_filename_extension(img_id_fname)
    object_id = f"{object_prefix}/{img_id}"

    if not img_id.endswith(img_format):
        object_id += f".{img_format}"

    print("oid", object_id)
    return object_id


def store_image(img, img_id_fname: str, object_prefix, img_format="png"):
    stream, byte_count = get_image_stream(img, img_format)

    object_id = get_object_id(img_id_fname, object_prefix, img_format)

    date = datetime(2025, 12, 1)
    retention = Retention(GOVERNANCE, date)

    resp: ObjectWriteResult = minio_client.put_object(MINIO_BASE_BUCKET, object_id, stream, byte_count,
                                                      content_type=f"image/{img_format}", legal_hold=MINIO_USE_OBJECT_LOCK,
                                                      retention=retention if MINIO_USE_OBJECT_LOCK else None)

    object_path = MINIO_BASE_BUCKET + "/" + object_id

    return object_path


def store_analysis_result(source_image, ann_image, device_id):
    o_id = str(uuid4())
    day_id = datetime.now().strftime('%Y%m%d')

    o_prefix = f"{device_id}/{day_id}"

    object_path = store_image(source_image, o_id, object_prefix=o_prefix)
    ann_object_path = store_image(ann_image, o_id + "_ann", object_prefix=o_prefix, img_format="jpeg")
    print(object_path, ann_object_path)
    return object_path, ann_object_path


# matplotlib.use("MacOSX")
class UploadImage(Resource):
    def post(self, device_id):

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
        # result = None
        ann_img, detections = result[0]
        #Image.fromarray(ann_img).save("./test.png", format="png")

        if USE_MINIO:
            img_path, ann_img_path = store_analysis_result(image, ann_img, device_id)
            return {'detections': detections.tolist(),
                    'image_urls': {
                        'source': img_path, 'ann_img': ann_img_path
                    }}
        return {'detections': detections.tolist()}


class TestEndpoint(Resource):
    def get(self):
        return {'data': 'alive and running!'}


api.add_resource(UploadImage, '/<string:device_id>')
api.add_resource(TestEndpoint, '/health')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5555)
