# Modified by Kamirul Kamirul
# Contact: kamirul.apr@gmail.com

# Original implementation by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
import cv2


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_xyxy_mpo(x):
    x_c, y_c, w, h, da, db = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h),
         da, db]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def mpo_to_4points(coord):
    xc = coord[0]
    yc = coord[1]
    ww = coord[2]
    hh = coord[3]
    da = coord[4]
    db = coord[5]

    p1x = xc + da
    p1y = yc - hh/2.0
    p2x = xc + ww/2.0
    p2y = yc - db
    p3x = xc - da
    p3y = yc + hh/2.0
    p4x = xc - ww/2.0
    p4y = yc + db

    return torch.tensor([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y])

def assign_absolute_angle(rbox):

    ang = rbox[:,4]
    ang[ang<0] = 1-abs(ang[ang<0])
    rbox[:,4]=ang
    return rbox
            


def xywha_to_4pts_targets(gt_boxes):
    nboxes = gt_boxes.shape[0]
    gt_boxes_4pts = torch.zeros((nboxes,8))

    for ibox in range (nboxes):

        gt_box = gt_boxes[ibox,:].cpu().numpy()

        mbox_cx = gt_box[0]
        mbox_cy = gt_box[1]
        mbox_w  = gt_box[2]
        mbox_h  = gt_box[3]
        angle   = gt_box[4]
        
        rect = ((mbox_cx, mbox_cy), (mbox_w, mbox_h), angle)
        pts4 = torch.tensor(cv2.boxPoints(rect))  # 4 x 2

        gt_boxes_4pts[ibox,:] = pts4.reshape(1,-1).squeeze(0)

    return gt_boxes_4pts 




# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area



def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)



# For XYRAB
def xyrab2xywha(xyrab):


    gx = xyrab[..., 0] 
    gy = xyrab[..., 1] 
    gr = xyrab[..., 2]
    ga = torch.deg2rad(xyrab[..., 3])
    gb = torch.deg2rad(xyrab[..., 4])


    x1 = gx + gr * torch.sin(ga)
    y1 = gy - gr * torch.cos(ga)
    x2 = gx + gr * torch.cos(gb)
    y2 = gy - gr * torch.sin(gb)
    x3 = gx - gr * torch.sin(ga)
    y3 = gy + gr * torch.cos(ga)
    x4 = gx - gr * torch.cos(gb)
    y4 = gy + gr * torch.sin(gb)


    ww = torch.sqrt(((x2-x1)**2) + ((y2-y1)**2) )
    hh = torch.sqrt(((x3-x2)**2) + ((y3-y2)**2) )
    

    y1y2 = y4-y1
    x1x2 = x4-x1


    deg90 = torch.tensor(90.0)
    deg180 = torch.tensor(180.0)

    theta = torch.rad2deg(torch.atan2(x1x2,y1y2))
    #theta = theta + deg90

    #theta[theta>180] = -(deg180 - (theta[theta>180] - deg180))
    #theta[theta<-180] = deg180 - (torch.abs(theta[theta<-180])-deg180)

    #print(theta)

    return torch.stack([gx, gy, ww, hh, theta], dim=-1)
    