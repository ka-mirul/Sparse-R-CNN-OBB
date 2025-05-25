# Written by Kamirul Kamirul
# Contact: kamirul.apr@gmail.com

import os
import numpy as np
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def get_rsdd_dataset_function(data_dir, phase):
    def dataset_function():
        return rsdd_directory_to_detectron_dataset(data_dir, phase)
    return dataset_function


def rsdd_directory_to_detectron_dataset(data_dir, phase):
    class_labels = ["ship"]
    image_dir = os.path.join(data_dir, 'JPEGImages')
    label_dir = os.path.join(data_dir, 'Annotations')

    # open imagesets file
    image_set_index_file = os.path.join((os.path.join(data_dir, 'ImageSets')), phase + '.txt')
    assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
    with open(image_set_index_file, 'r') as f:
        lines = f.readlines()
    files = [line.strip() for line in lines]

    images = []
    classes = []

    for i, filename in enumerate(files):
        image_name = os.path.join(image_dir,  filename +'.jpg')
        label_name = os.path.join(label_dir,  filename +'.xml')

        if not ((os.path.isfile(image_name)) and (os.path.isfile(label_name))):
            continue

        annotations = []
        target = ET.parse(label_name).getroot()

        for obj in target.iter('size'):
            imageW = int(obj.find('width').text)
            imageH = int(obj.find('height').text)


        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text)
            robj = obj.find('robndbox')
            for values in robj:
                mbox_cx = float(robj.find('cx').text)  # rbox
                mbox_cy = float(robj.find('cy').text)
                mbox_w = float(robj.find('w').text)
                mbox_h = float(robj.find('h').text)
                angle = float(robj.find('angle').text)*180/np.pi


            if (angle>0):
                angle = 90 - angle
            else:
                angle = -(90 + angle)

            
            annotations.append({
                "bbox_mode": 4,  # Oriented bounding box (cx, cy, w, h, a)
                "category_id": 0,
                "bbox": (mbox_cx, mbox_cy, mbox_w, mbox_h, angle)
            })

        images.append({
                "id": int(i),
                "file_name": image_name,
                "annotations": annotations,
                "width" : imageW,
                "height" : imageH,
                })
    return images
