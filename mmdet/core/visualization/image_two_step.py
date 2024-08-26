# Copyright (c) OpenMMLab. All rights reserved.
import sys
import os
import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as mask_util
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from mmdet.core.evaluation.panoptic_utils import INSTANCE_OFFSET
from ..mask.structures import bitmap_to_polygon
from ..utils import mask2ndarray
from .palette import get_palette, palette_val

import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import wandb
import warnings
from .models_fgvcpim.builder import MODEL_GETTER
from .data_fgvcpim.dataset import build_loader
from .utils_fgvcpim.costom_logger import timeLogger
from .utils_fgvcpim.config_utils import load_yaml, build_record_folder, get_args
from typing import Union
def top_k_corrects(preds: torch.Tensor, labels: torch.Tensor, tops: list = [1, 3, 5]):
    """
    preds: [B, C] (C is num_classes)
    labels: [B, ]
    """
    if preds.device != torch.device('cpu'):
        preds = preds.cpu()
    if labels.device != torch.device('cpu'):
        labels = labels.cpu()
    tmp_cor = 0
    corrects = {"top-"+str(x):0 for x in tops}
    sorted_preds = torch.sort(preds, dim=-1, descending=True)[1]
    for i in range(tops[-1]):
        tmp_cor += sorted_preds[:, i].eq(labels).sum().item()
        # records
        if "top-"+str(i+1) in corrects:
            corrects["top-"+str(i+1)] = tmp_cor
    return corrects

@torch.no_grad()
def _cal_evalute_metric(corrects: dict, 
                        total_samples: dict,
                        logits: torch.Tensor, 
                        labels: torch.Tensor, 
                        this_name: str,
                        scores: Union[list, None] = None, 
                        score_names: Union[list, None] = None):
    
    tmp_score = torch.softmax(logits, dim=-1)
    tmp_corrects = top_k_corrects(tmp_score, labels, tops=[1, 3]) # return top-1, top-3, top-5 accuracy
    
    ### each layer's top-1, top-3 accuracy
    for name in tmp_corrects:
        eval_name = this_name + "-" + name
        if eval_name not in corrects:
            corrects[eval_name] = 0
            total_samples[eval_name] = 0
        corrects[eval_name] += tmp_corrects[name]
        total_samples[eval_name] += labels.size(0)
    
    if scores is not None:
        scores.append(tmp_score)
    if score_names is not None:
        score_names.append(this_name)


def step2_inference(x,model1):
    cv2.imwrite('temp/sample/'+"test.png", x)
    val_loader = build_loader("temp/")

    corrects = {}
    total_samples = {}
    results = []
    with torch.no_grad():
        """ accumulate """
        for batch_id, (ids, datas, labels) in enumerate(val_loader):

            score_names = []
            scores = []
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            datas = datas.to(device)
            outs = model1(datas)

            # if args.use_fpn and (0 < args.highest < 5):
            #     this_name = "layer" + str(args.highest)
            #     _cal_evalute_metric(corrects, total_samples, outs[this_name].mean(1), labels, this_name, scores, score_names)

            if True:
                this_name = "combiner"
                _cal_evalute_metric(corrects, total_samples, outs["comb_outs"], labels, this_name, scores, score_names)

            # _average_top_k_result(corrects, total_samples, scores, labels)

            # for i in range(scores[0].shape[0]):
            #     #print(test_loader.dataset.data_infos[ids[i].item()]['path'])
            #     #print(int(labels[i].item()))   
            #     print(int(scores[0][i].argmax().item()))
            #     #print(scores[0][i][scores[0][i].argmax().item()].item())   

    return int(scores[0][0].argmax().item())
__all__ = [
    'color_val_matplotlib', 'draw_masks', 'draw_bboxes', 'draw_labels',
    'imshow_det_bboxes', 'imshow_gt_det_bboxes'
]

EPS = 1e-2


def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples.

    Args:
        color (:obj`Color` | str | tuple | int | ndarray): Color inputs.

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


def _get_adaptive_scales(areas, min_area=800, max_area=30000):
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``'min_area'``, the scale is 0.5 while the area is larger than
    ``'max_area'``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Default: 800.
        max_area (int): Upper bound areas for adaptive scales.
            Default: 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


def _get_bias_color(base, max_dist=30):
    """Get different colors for each masks.

    Get different colors for each masks by adding a bias
    color to the base category color.
    Args:
        base (ndarray): The base category color with the shape
            of (3, ).
        max_dist (int): The max distance of bias. Default: 30.

    Returns:
        ndarray: The new color for a mask with the shape of (3, ).
    """
    new_color = base + np.random.randint(
        low=-max_dist, high=max_dist + 1, size=3)
    return np.clip(new_color, 0, 255, new_color)


def draw_bboxes(ax, bboxes, color='g', alpha=0.8, thickness=2):
    """Draw bounding boxes on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        bboxes (ndarray): The input bounding boxes with the shape
            of (n, 4).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.

    Returns:
        matplotlib.Axes: The result axes.
    """
    polygons = []
    for i, bbox in enumerate(bboxes):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
    p = PatchCollection(
        polygons,
        facecolor='none',
        edgecolors=color,
        linewidths=thickness,
        alpha=alpha)
    ax.add_collection(p)

    return ax


def draw_labels(ax,
                labels,
                positions,
                scores=None,
                class_names=None,
                color='w',
                font_size=8,
                scales=None,
                horizontal_alignment='left'):
    """Draw labels on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        labels (ndarray): The labels with the shape of (n, ).
        positions (ndarray): The positions to draw each labels.
        scores (ndarray): The scores for each labels.
        class_names (list[str]): The class names.
        color (list[tuple] | matplotlib.color): The colors for labels.
        font_size (int): Font size of texts. Default: 8.
        scales (list[float]): Scales of texts. Default: None.
        horizontal_alignment (str): The horizontal alignment method of
            texts. Default: 'left'.

    Returns:
        matplotlib.Axes: The result axes.
    """
    for i, (pos, label) in enumerate(zip(positions, labels)):
        if label==98:
            break
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        # if scores is not None:
        #     label_text += f'|{scores[i]:.02f}' modify 
        text_color = color[i] if isinstance(color, list) else color

        font_size_mask = font_size if scales is None else font_size * scales[i]
        ax.text(
            pos[0],
            pos[1],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=font_size_mask,
            verticalalignment='top',
            horizontalalignment=horizontal_alignment)

    return ax


def draw_masks(ax, img, masks, color=None, with_edge=True, alpha=0.8):
    """Draw masks on the image and their edges on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        img (ndarray): The image with the shape of (3, h, w).
        masks (ndarray): The masks with the shape of (n, h, w).
        color (ndarray): The colors for each masks with the shape
            of (n, 3).
        with_edge (bool): Whether to draw edges. Default: True.
        alpha (float): Transparency of bounding boxes. Default: 0.8.

    Returns:
        matplotlib.Axes: The result axes.
        ndarray: The result image.
    """
    taken_colors = set([0, 0, 0])
    if color is None:
        random_colors = np.random.randint(0, 255, (masks.size(0), 3))
        color = [tuple(c) for c in random_colors]
        color = np.array(color, dtype=np.uint8)
    polygons = []
    for i, mask in enumerate(masks):
        if with_edge:
            contours, _ = bitmap_to_polygon(mask)
            polygons += [Polygon(c) for c in contours]

        color_mask = color[i]
        while tuple(color_mask) in taken_colors:
            color_mask = _get_bias_color(color_mask)
        taken_colors.add(tuple(color_mask))

        mask = mask.astype(bool)
        img[mask] = img[mask] * (1 - alpha) + color_mask * alpha

    p = PatchCollection(
        polygons, facecolor='none', edgecolors='w', linewidths=1, alpha=0.8)
    ax.add_collection(p)

    return ax, img


def imshow_det_bboxes(img,
                      bboxes=None,
                      labels=None,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=8,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray | None): Masks, shaped (n,h,w) or None.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.
        bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        text_color (list[tuple] | tuple | str | None): Colors of texts.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        mask_color (list[tuple] | tuple | str | None, optional): Colors of
           masks. If a single color is given, it will be applied to all
           classes. The tuple of color should be in RGB order.
           Default: None.
        thickness (int): Thickness of lines. Default: 2.
        font_size (int): Font size of texts. Default: 13.
        show (bool): Whether to show the image. Default: True.
        win_name (str): The window name. Default: ''.
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes is None or bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    assert segms is None or segms.shape[0] == labels.shape[0], \
        'segms.shape[0] and labels.shape[0] should have the same length.'
    assert segms is not None or bboxes is not None, \
        'segms and bboxes should not be None at the same time.'
    #print(img)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ### = = = =  Model = = = =
    #tlogger.print("Building Model....")
    model1 = MODEL_GETTER['swin-t'](
        use_fpn=True,
        fpn_size=1536,
        use_selection=True,
        num_classes=200,
        num_selects= {
            "layer1": 2048,
            "layer2": 512,
            "layer3": 128,
            "layer4": 32,
            },
        use_combiner=True,
    )  # about return_nodes, we use our default setting
    checkpoint = torch.load("step2_checkpoint.pt", map_location=torch.device('cpu'))
    model1.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']
    model1=model1.to(device)
    #print(model)
    model1.eval()




    img = mmcv.imread(img).astype(np.uint8)
    crop_image=img
    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]
    #print(bboxes )
    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    max_label = int(max(labels) if len(labels) > 0 else 0)
    text_palette = palette_val(get_palette(text_color, max_label + 1))
    text_colors = [text_palette[label] for label in labels]
    num_bboxes = 0
    if bboxes is not None:
        num_bboxes = bboxes.shape[0]
        # for  i,bbox in enumerate(bboxes):
            # print(i,int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
            # cv2.imwrite('test'+"/"+str(i)+".png", crop_image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])])
        bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        colors = [bbox_palette[label] for label in labels[:num_bboxes]]
        draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)

        horizontal_alignment = 'left'
        positions = bboxes[:, :2].astype(np.int32) + thickness
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)
        scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
        #crop = img[]
        for  i,bbox in enumerate(bboxes):
            x=crop_image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
            labels[i]=step2_inference(x,model1)
        class_names=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","52","53","54","55","56","57","58","59","60","61","62","63","64","65","66","67","68","69","70","71","72","73","74","75","76","77","78","79","80","81","82","83","84","85","86","87","88","89","90","91","92","93","94","95","96","97","-1"]
        draw_labels(
            ax,
            labels[:num_bboxes],
            positions,
            scores=scores,
            class_names=class_names,
            color=text_colors,
            font_size=font_size,
            scales=scales,
            horizontal_alignment=horizontal_alignment)

    if segms is not None:
        mask_palette = get_palette(mask_color, max_label + 1)
        colors = [mask_palette[label] for label in labels]
        colors = np.array(colors, dtype=np.uint8)
        draw_masks(ax, img, segms, colors, with_edge=True)

        if num_bboxes < segms.shape[0]:
            segms = segms[num_bboxes:]
            horizontal_alignment = 'center'
            areas = []
            positions = []
            for mask in segms:
                _, _, stats, centroids = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8), connectivity=8)
                largest_id = np.argmax(stats[1:, -1]) + 1
                positions.append(centroids[largest_id])
                areas.append(stats[largest_id, -1])
            areas = np.stack(areas, axis=0)
            scales = _get_adaptive_scales(areas)
            draw_labels(
                ax,
                labels[num_bboxes:],
                positions,
                class_names=class_names,
                color=text_colors,
                font_size=font_size,
                scales=scales,
                horizontal_alignment=horizontal_alignment)

    plt.imshow(img)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    if sys.platform == 'darwin':
        width, height = canvas.get_width_height(physical=True)
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img


def imshow_gt_det_bboxes(img,
                         annotation,
                         result,
                         class_names=None,
                         score_thr=0,
                         gt_bbox_color=(61, 102, 255),
                         gt_text_color=(200, 200, 200),
                         gt_mask_color=(61, 102, 255),
                         det_bbox_color=(241, 101, 72),
                         det_text_color=(200, 200, 200),
                         det_mask_color=(241, 101, 72),
                         thickness=2,
                         font_size=13,
                         win_name='',
                         show=True,
                         wait_time=0,
                         out_file=None,
                         overlay_gt_pred=True):
    """General visualization GT and result function.

    Args:
      img (str | ndarray): The image to be displayed.
      annotation (dict): Ground truth annotations where contain keys of
          'gt_bboxes' and 'gt_labels' or 'gt_masks'.
      result (tuple[list] | list): The detection result, can be either
          (bbox, segm) or just bbox.
      class_names (list[str]): Names of each classes.
      score_thr (float): Minimum score of bboxes to be shown. Default: 0.
      gt_bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (61, 102, 255).
      gt_text_color (list[tuple] | tuple | str | None): Colors of texts.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (200, 200, 200).
      gt_mask_color (list[tuple] | tuple | str | None, optional): Colors of
          masks. If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (61, 102, 255).
      det_bbox_color (list[tuple] | tuple | str | None):Colors of bbox lines.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (241, 101, 72).
      det_text_color (list[tuple] | tuple | str | None):Colors of texts.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (200, 200, 200).
      det_mask_color (list[tuple] | tuple | str | None, optional): Color of
          masks. If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (241, 101, 72).
      thickness (int): Thickness of lines. Default: 2.
      font_size (int): Font size of texts. Default: 13.
      win_name (str): The window name. Default: ''.
      show (bool): Whether to show the image. Default: True.
      wait_time (float): Value of waitKey param. Default: 0.
      out_file (str, optional): The filename to write the image.
          Default: None.
      overlay_gt_pred (bool): Whether to plot gts and predictions on the
       same image. If False, predictions and gts will be plotted on two same
       image which will be concatenated in vertical direction. The image
       above is drawn with gt, and the image below is drawn with the
       prediction result. Default: True.

    Returns:
        ndarray: The image with bboxes or masks drawn on it.
    """
    assert 'gt_bboxes' in annotation
    assert 'gt_labels' in annotation
    assert isinstance(result, (tuple, list, dict)), 'Expected ' \
        f'tuple or list or dict, but get {type(result)}'

    gt_bboxes = annotation['gt_bboxes']
    gt_labels = annotation['gt_labels']
    gt_masks = annotation.get('gt_masks', None)
    if gt_masks is not None:
        gt_masks = mask2ndarray(gt_masks)

    gt_seg = annotation.get('gt_semantic_seg', None)
    if gt_seg is not None:
        pad_value = 255  # the padding value of gt_seg
        sem_labels = np.unique(gt_seg)
        all_labels = np.concatenate((gt_labels, sem_labels), axis=0)
        all_labels, counts = np.unique(all_labels, return_counts=True)
        stuff_labels = all_labels[np.logical_and(counts < 2,
                                                 all_labels != pad_value)]
        stuff_masks = gt_seg[None] == stuff_labels[:, None, None]
        gt_labels = np.concatenate((gt_labels, stuff_labels), axis=0)
        gt_masks = np.concatenate((gt_masks, stuff_masks.astype(np.uint8)),
                                  axis=0)
        # If you need to show the bounding boxes,
        # please comment the following line
        # gt_bboxes = None

    img = mmcv.imread(img)

    img_with_gt = imshow_det_bboxes(
        img,
        gt_bboxes,
        gt_labels,
        gt_masks,
        class_names=class_names,
        bbox_color=gt_bbox_color,
        text_color=gt_text_color,
        mask_color=gt_mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=False)

    if not isinstance(result, dict):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None

        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            segms = mask_util.decode(segms)
            segms = segms.transpose(2, 0, 1)
    else:
        assert class_names is not None, 'We need to know the number ' \
                                        'of classes.'
        VOID = len(class_names)
        bboxes = None
        pan_results = result['pan_results']
        # keep objects ahead
        ids = np.unique(pan_results)[::-1]
        legal_indices = ids != VOID
        ids = ids[legal_indices]
        labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
        segms = (pan_results[None] == ids[:, None, None])

    if overlay_gt_pred:
        img = imshow_det_bboxes(
            img_with_gt,
            bboxes,
            labels,
            segms=segms,
            class_names=class_names,
            score_thr=score_thr,
            bbox_color=det_bbox_color,
            text_color=det_text_color,
            mask_color=det_mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)
    else:
        img_with_det = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms=segms,
            class_names=class_names,
            score_thr=score_thr,
            bbox_color=det_bbox_color,
            text_color=det_text_color,
            mask_color=det_mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=False)
        img = np.concatenate([img_with_gt, img_with_det], axis=0)

        plt.imshow(img)
        if show:
            if wait_time == 0:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)
        plt.close()

    return img
