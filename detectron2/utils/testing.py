import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.data import DatasetCatalog
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.structures import Boxes
from detectron2.utils.file_io import PathManager


"""
Internal utilities for tests. Don't use except for writing tests.
"""


def get_model_no_weights(config_path):
    """
    Like model_zoo.get, but do not load any weights (even pretrained)
    """
    cfg = model_zoo.get_config(config_path)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    return build_model(cfg)


def random_boxes(num_boxes, max_coord=100, device="cpu"):
    """
    Create a random Nx4 boxes tensor, with coordinates < max_coord.
    """
    boxes = torch.rand(num_boxes, 4, device=device) * (max_coord * 0.5)
    boxes.clamp_(min=1.0)  # tiny boxes cause numerical instability in box regression
    # Note: the implementation of this function in torchvision is:
    # boxes[:, 2:] += torch.rand(N, 2) * 100
    # but it does not guarantee non-negative widths/heights constraints:
    # boxes[:, 2] >= boxes[:, 0] and boxes[:, 3] >= boxes[:, 1]:
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def get_sample_coco_image(tensor=True):
    """
    Args:
        tensor (bool): if True, returns 3xHxW tensor.
            else, returns a HxWx3 numpy array.

    Returns:
        an image, in BGR color.
    """
    try:
        file_name = DatasetCatalog.get("coco_2017_train")[0]["file_name"]
        if not PathManager.exists(file_name):
            raise FileNotFoundError()
    except IOError:
        # for public CI to run
        file_name = "http://images.cocodataset.org/train2017/000000000009.jpg"
    ret = read_image(file_name, format="BGR")
    if tensor:
        ret = torch.from_numpy(np.ascontiguousarray(ret.transpose(2, 0, 1)))
    return ret


def assert_instances_allclose(input, other, rtol=1e-5, msg=""):
    """
    Args:
        input, other (Instances):
    """
    if not msg:
        msg = "Two Instances are different! "
    else:
        msg = msg.rstrip() + " "
    assert input.image_size == other.image_size, (
        msg + f"image_size is {input.image_size} vs. {other.image_size}!"
    )
    fields = sorted(input.get_fields().keys())
    fields_other = sorted(other.get_fields().keys())
    assert fields == fields_other, msg + f"Fields are {fields} vs {fields_other}!"

    for f in fields:
        val1, val2 = input.get(f), other.get(f)
        if isinstance(val1, Boxes):
            # boxes in the range of O(100) and can have a larger tolerance
            assert torch.allclose(val1.tensor, val2.tensor, atol=100 * rtol), (
                msg + f"Field {f} differs too much!"
            )
        elif isinstance(val1, torch.Tensor):
            if val1.dtype.is_floating_point:
                mag = torch.abs(val1).max().cpu().item()
                assert torch.allclose(val1, val2, atol=mag * rtol), (
                    msg + f"Field {f} differs too much!"
                )
            else:
                assert torch.equal(val1, val2), msg + f"Field {f} is different!"
        else:
            raise ValueError(f"Don't know how to compare type {type(val1)}")
