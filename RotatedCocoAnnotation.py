from sahi.utils.coco import CocoAnnotation
import copy
import logging
import os
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional, Set, Union

import numpy as np
from shapely import MultiPolygon
from shapely.validation import make_valid
from tqdm import tqdm

from sahi.utils.file import is_colab, load_json, save_json
from sahi.utils.shapely import ShapelyAnnotation, box, get_shapely_multipolygon

class CocoAnnotationOBB(CocoAnnotation):

    def __init__(
        self,
        segmentation=None,
        bbox=None,
        category_id=None,
        category_name=None,
        image_id=None,
        iscrowd=0,
    ):
        

        if bbox is None and segmentation is None:
            raise ValueError("you must provide a bbox or polygon")

        self._segmentation = segmentation
        self._category_id = category_id
        self._category_name = category_name
        self._image_id = image_id
        self._iscrowd = iscrowd
        self._bbox = bbox

        
        
        if self._segmentation:
            shapely_annotation = ShapelyAnnotation.from_coco_segmentation(segmentation=self._segmentation)
        else:
            shapely_annotation = ShapelyAnnotation.from_coco_bbox(bbox=bbox)
        self._shapely_annotation = shapely_annotation

        
    @property
    def bbox(self):
        """
        Returns coco formatted bbox of the annotation as [xmin, ymin, width, height]
        """

        return self._bbox

        #return self._shapely_annotation.to_xywh()
    
    @property
    def area(self):
        """
        Returns coco formatted bbox of the annotation as [xmin, ymin, width, height]
        """

        return self._bbox[2]* self._bbox[3]

        
