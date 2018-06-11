# -*- coding: utf-8 -*-
# file: yolov3.py
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
the main process of yolo-v3
here we are mainly follow these steps:

DarkNet53 feature extraction ->
predict dx, dy, dw, dh, get very cell predict object and probability ->
get final result.

For train process here are some difficult part.

"""
import tensorflow as tf
import numpy as np

from .config import global_config
from .darknet53 import darknet53


class YoloV3(object):
    def __init__(self):
        pass

    def _yolo_block(self, inputs, filters):
        inputs = darknet53.conv2d_fixed_padding(inputs, filters, 1)
        inputs = darknet53.conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = darknet53.conv2d_fixed_padding(inputs, filters, 1)
        inputs = darknet53.conv2d_fixed_padding(inputs, filters * 2, 3)
        inputs = darknet53.conv2d_fixed_padding(inputs, filters, 1)
        route = inputs
        inputs = darknet53.conv2d_fixed_padding(inputs, filters * 2, 3)
        return route, inputs

    @staticmethod
    def _get_size(shape, data_format):
        if len(shape) == 4:
            shape = shape[1:]
        return shape[1:3] if data_format == 'NCHW' else shape[0:2]

    def _detection_layer(self, inputs, num_classes, anchors, img_size, data_format):
        """
        this is the main logic in Yolo
        :param inputs:
        :param num_classes:
        :param anchors:
        :param img_size:
        :param data_format:
        :return:
        """
        if not isinstance(anchors, list):
            raise TypeError('anchors must be a list object')
        else:
            num_anchors = len(anchors)
            predictions = tf.layers.conv2d(inputs, num_anchors * (5 + num_classes), 1, strides=1,
                                           activation=None, bias_initializer=tf.zeros_initializer,
                                           kernel_initializer=None)
            shape = predictions.get_shape().as_list()
            grid_size = self._get_size(shape, data_format)
            dim = grid_size[0] * grid_size[1]
            bbox_attrs = 5 + num_classes
            if data_format == 'NCHW':
                predictions = tf.reshape(predictions, [-1, num_anchors * bbox_attrs, dim])
                predictions = tf.transpose(predictions, [0, 2, 1])
            predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])
            stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
            anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]
            box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes], axis=-1)
            box_centers = tf.nn.sigmoid(box_centers)
            confidence = tf.nn.sigmoid(confidence)

            grid_x = tf.range(grid_size[0], dtype=tf.float32)
            grid_y = tf.range(grid_size[1], dtype=tf.float32)
            a, b = tf.meshgrid(grid_x, grid_y)

            x_offset = tf.reshape(a, (-1, 1))
            y_offset = tf.reshape(b, (-1, 1))

            x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
            x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

            box_centers += x_y_offset
            box_centers *= stride

            anchors = tf.tile(anchors, [dim, 1])
            box_sizes = tf.exp(box_sizes) * anchors
            box_sizes *= stride

            detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)

            classes = tf.nn.sigmoid(classes)
            predictions = tf.concat([detections, classes], axis=-1)
            return predictions

    @staticmethod
    def _up_sample(inputs, out_shape, data_format='NCHW'):
        inputs = darknet53.fix_padding(inputs, 3, 'NHWC', mode='SYMMETRIC')
        if data_format == 'NCHW':
            inputs = tf.transpose(inputs, [0, 2, 3, 1])
        if data_format == 'NCHW':
            height = out_shape[3]
            width = out_shape[2]
        else:
            height = out_shape[2]
            width = out_shape[1]
        new_height = height + 4
        new_width = width + 4

        inputs = tf.image.resize_bilinear(inputs, (new_height, new_width))
        inputs = inputs[:, 2:-2, 2:-2, :]
        if data_format == 'NCHW':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        inputs = tf.identity(inputs, name='up_sampled')
        return inputs

    def yolo_v3_pipeline(self, inputs, num_classes, img_size, data_format):
        with tf.variable_scope('darknet-53'):
            route_1, route_2, inputs = darknet53.build_model(inputs)
        with tf.variable_scope('yolo-v3'):
            route, inputs = self._yolo_block(inputs, 512)
            detect_1 = self._detection_layer(inputs, num_classes, global_config.anchors,
                                             img_size, data_format)
            inputs = darknet53.conv2d_fixed_padding(route, 256, 1)
            up_sample_size = route_2.get_shape().as_list()
            inputs = self._up_sample(inputs, up_sample_size, data_format)
            inputs = tf.concat([inputs, route_2], axis=1 if data_format == 'NCHW' else 3)
            route, inputs = self._yolo_block(inputs, 256)
            detect_2 = self._detection_layer(inputs, num_classes, global_config.anchors, img_size, data_format)
            detect_2 = tf.identity(detect_2, name='detect_2')
            inputs = darknet53.conv2d_fixed_padding(route, 128, 1)
            up_sample_size = route_1.get_shape().as_list()
            inputs = self._up_sample(inputs, up_sample_size, data_format)
            inputs = tf.concat([inputs, route_1], axis=1 if data_format == 'NCHW' else 3)

            _, inputs = self._yolo_block(inputs, 128)
            detect_3 = self._detection_layer(inputs, num_classes, global_config.anchors, img_size, data_format)
            detect_3 = tf.identity(detect_3, name='detect_3')

            detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
            return detections

    @staticmethod
    def load_weights_from_yolo_origin(var_list, file_name):
        """
        this method just for testing to see our
        pipeline is work well or not
        :param var_list
        :param file_name:
        :return:
        """
        with open(file_name, 'rb') as fp:
            _ = np.fromfile(fp, dtype=np.int32, count=5)
            weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        i = 0
        assign_ops = []
        # var_list = []
        while i < len(var_list) - 1:
            var_1 = var_list[i]
            var_2 = var_list[i+1]
            if 'Conv' in var_1.name.split('/')[-2]:
                if 'BatchNorm' in var_2.name.split('/')[-2]:
                    gama, beta, mean, var = var_list[i+1:i+5]
                    batch_norm_vars = [beta, gama, mean, var]
                    for var in batch_norm_vars:
                        shape = var.shape.as_list()
                        num_params = np.prod(shape)
                        var_weights = weights[ptr: ptr + num_params].reshape(shape)
                        ptr += num_params
                        assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                        i += 4
                elif 'Conv' in var_2.name.split('/')[-2]:
                    bias = var_2
                    bias_shape = bias.shape.as_list()
                    bias_params = np.prod(bias_shape)
                    bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                    ptr += bias_params
                    assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                    i += 1
                shape = var_1.shape.as_list()
                num_params = np.prod(shape)
                var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
                i += 1
        return assign_ops

    @staticmethod
    def detections_boxes(detections):
        """
        get normal boxes
        :param detections:
        :return:
        """
        center_x, center_y, width, height, attrs = tf.split(detections, [1, 1, 1, 1, -1], axis=-1)
        w2 = width / 2
        h2 = height / 2
        x0 = center_x - w2
        y0 = center_y - h2
        x1 = center_x + w2
        y1 = center_y + h2

        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        detections = tf.concat([boxes, attrs], axis=-1)
        return detections


