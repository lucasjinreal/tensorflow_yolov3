# -*- coding: utf-8 -*-
# file: config.py
# author: JinTian
# time: 2018/6/11 10:32 AM
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
fine-tuned params by Yolo-V3 already
we can just using this

"""


class Config(object):

    def __init__(self):
        """
        Yolo-V3 normalize the input into 0..1
        most CNN do batch-normalization right after Convolution, d't using
        bias and using Leaky_ReLU as activation function
        """

        self.batch_norm_decay = 0.9
        self.batch_norm_epsilon = 1e-05
        self.l_ReLU = 0.1

        self.anchors = [
            (10, 13),
            (16, 30),
            (33, 23),
            (30, 61), (62, 45),
            (59, 119), (116, 90),
            (156, 198), (373, 326)
        ]

        self.nms_cf_threshold = 0.5
        self.nms_iou_threshold = 0.4

        self.coco_names = [
            'bus',
            'dinasour'
        ]

global_config = Config()