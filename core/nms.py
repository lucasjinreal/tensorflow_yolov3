# -*- coding: utf-8 -*-
# file: nms.py
# author: JinTian
# time: 2018/6/11 8:42 PM
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
A quite normal work of nms
"""
import numpy as np
from .config import global_config


class NMS(object):

    def __init__(self):
        pass

    @staticmethod
    def _iou(box1, box2):
        b1_x0, b1_y0, b1_x1, b1_y1 = box1
        b2_x0, b2_y0, b2_x1, b2_y1 = box2

        int_x0 = max(b1_x0, b2_x0)
        int_y0 = max(b1_y0, b2_y0)
        int_x1 = min(b1_x1, b2_x1)
        int_y1 = min(b1_y1, b2_y1)

        int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

        b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
        b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

        iou = int_area / (b1_area + b2_area - int_area + 1e-05)
        return iou

    def nms(self, predictions_with_boxes):
        """
        do nms
        :param predictions_with_boxes:
        :param confidence_threshold:
        :param iou_threshold:
        :return:
        """
        conf_mask = np.expand_dims((predictions_with_boxes[:, :, 4] > global_config.nms_cf_threshold), -1)
        predictions = predictions_with_boxes * conf_mask

        result = {}
        for i, image_pred in enumerate(predictions):
            shape = image_pred.shape
            non_zero_idxs = np.nonzero(image_pred)
            image_pred = image_pred[non_zero_idxs]
            image_pred = image_pred.reshape(-1, shape[-1])

            bbox_attrs = image_pred[:, :5]
            classes = image_pred[:, 5:]
            classes = np.argmax(classes, axis=-1)

            unique_classes = list(set(classes.reshape(-1)))

            for cls in unique_classes:
                cls_mask = classes == cls
                cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
                cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
                cls_scores = cls_boxes[:, -1]
                cls_boxes = cls_boxes[:, :-1]

                while len(cls_boxes) > 0:
                    box = cls_boxes[0]
                    score = cls_scores[0]
                    if not cls in result:
                        result[cls] = []
                    result[cls].append((box, score))
                    cls_boxes = cls_boxes[1:]
                    ious = np.array([self._iou(box, x) for x in cls_boxes])
                    iou_mask = ious < global_config.nms_iou_threshold
                    cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                    cls_scores = cls_scores[np.nonzero(iou_mask)]

        return result

nms = NMS()


