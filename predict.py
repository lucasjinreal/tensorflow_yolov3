# -*- coding: utf-8 -*-
# file: predict.py
# author: JinTian
# time: 2018/6/11 10:06 AM
# Copyright 2018 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""
testing the Yolo-v3 detector
we now support for predict directly from trained weight from original author
or predict from your trained from scratch
previous are default predict method
"""
from core.yolov3 import YoloV3
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
from core.nms import nms
import json
import sys


def load_coco_names():
    label_map_dict = json.load('data/mscoco_label_map.json')
    names = dict()
    for i in label_map_dict:
        names[i['id']] = i['display_name']
    return names


def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


def draw_boxes(boxes, img, cls_names, detection_size):
    draw = ImageDraw.Draw(img)

    for cls, bboxs in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
            draw.rectangle(box, outline=color)
            draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)


def predict_from_origin_weights(img_f):
    img_size = (416, 416)
    img = Image.open(img_f)
    img_resized = img.resize(img_size)

    classes = load_coco_names()
    print(classes[0])

    inputs = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], 3])

    yolo_v3 = YoloV3()
    with tf.variable_scope('detector'):
        detections = yolo_v3.yolo_v3_pipeline(inputs, 80, img_size, data_format='NCHW')
        load_ops = yolo_v3.load_weights_from_yolo_origin(tf.global_variables(scope='model'), 'yolov3.weights')

    boxes = yolo_v3.detections_boxes(detections)
    with tf.Session() as sess:
        sess.run(load_ops)
        detected_boxes = sess.run(boxes, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})
    final_boxes = nms.nms(detected_boxes)
    draw_boxes(final_boxes, img, classes, img_size)
    img.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Pls provide a image to predict')
    else:
        print('Predicting from {}'.format(sys.argv[1]))
        predict_from_origin_weights(sys.argv[1])


