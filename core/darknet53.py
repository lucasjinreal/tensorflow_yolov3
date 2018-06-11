# -*- coding: utf-8 -*-
# file: darknet53.py
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
this is the implementation of DarkNet53 using in Yolo-V3

using core tf.layers.conv2d implement main network architecture, slim is deprecated


"""
import tensorflow as tf
import cv2


class DarkNet53(object):

    def __init__(self):
        pass

    @staticmethod
    def fix_padding(inputs, kernel_size, mode='CONSTANT'):
        """
        fixed padding whatever inputs is
        this code is get from ResNet from models repo
        :param inputs:
        :param kernel_size:
        :param mode:
        :return:
        """
        pad_total = kernel_size - 1
        pad_start = pad_total // 2
        pad_end = pad_total - pad_start
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_start, pad_end], [pad_start, pad_end], [0, 0]], mode=mode)
        return padded_inputs

    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides=1):
        if strides > 1:
            inputs = self.fix_padding(inputs, kernel_size)
        inputs = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                                  strides=strides, padding=('SAME' if strides == 1 else 'VALID'))
        return inputs

    def _darknet53_block(self, inputs, filters):
        shortcut = inputs
        # 2 layer convolution in DarkNet53, first is 1x1, next is 3x3 with fixed padding
        # only filters are various
        inputs = self.conv2d_fixed_padding(inputs, filters, 1)
        inputs = self.conv2d_fixed_padding(inputs, filters * 2, 3)

        # residual operation, why plus? just concat
        inputs += shortcut
        return inputs

    def build_model(self, inputs):
        """
        main process of building DarNet53
        :return:
        """
        # 1th part
        inputs = self.conv2d_fixed_padding(inputs, 32, 3)
        inputs = self.conv2d_fixed_padding(inputs, 64, 3, strides=2)
        inputs = self._darknet53_block(inputs, 32)

        # connector 1 - 128
        inputs = self.conv2d_fixed_padding(inputs, 128, 3, strides=2)

        # 2nd part
        for i in range(2):
            inputs = self._darknet53_block(inputs, 64)

        # connector 2 - 256
        inputs = self.conv2d_fixed_padding(inputs, 256, 3, strides=2)

        # 3rd part
        for i in range(8):
            inputs = self._darknet53_block(inputs, 128)

        # connector 3 - 512
        route_1 = inputs
        inputs = self.conv2d_fixed_padding(inputs, 512, 3, strides=2)

        # 4th
        for i in range(8):
            inputs = self._darknet53_block(inputs, 256)

        # connector 4 - 1024
        route_2 = inputs
        inputs = self.conv2d_fixed_padding(inputs, 1024, 3, strides=2)

        # 5th
        for i in range(4):
            inputs = self._darknet53_block(inputs, 512)
        # original DartNet53 have a more average pooling layer, and a soft-max, but we are not using
        # for classify, so just drop it
        return route_1, route_2, inputs


darknet53 = DarkNet53()