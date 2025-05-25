# Modified by Kamirul Kamirul
# Contact: kamirul.apr@gmail.com

# Original implementation by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen

from detectron2.structures import BoxMode

__all__ = ["SparseRCNNOBBDatasetMapper"]


def rotate_bbox(annotation, transforms):
    annotation["bbox"] = transforms.apply_rotated_box(
        np.asarray([annotation['bbox']]))[0]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS
    return annotation


def get_shape_augmentations(cfg):

    return [
        T.ResizeShortestEdge(short_edge_length=(
        128,800), max_size=1333, sample_style='range'),
        T.RandomFlip(),
    ]




class SparseRCNNOBBDatasetMapper:

    def __init__(self, cfg, is_train=True):
        
        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        self.cfg = cfg
        

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        image, image_transforms = T.apply_transform_gens(
            get_shape_augmentations(self.cfg), image)
        
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        annotations = [
            rotate_bbox(obj, image_transforms)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances_rotated(
            annotations, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

