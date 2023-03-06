import os
import gc
import glob
import time
import random
import imageio
import logging
from functools import wraps

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as torch_utils

from postprocess import SegDetectorRepresenter

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


def setup_determinism(seed=42):
    """
    https://github.com/pytorch/pytorch/issues/7068#issuecomment-487907668
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_device(batch, device='cuda'):
    new_batch = []

    for ele in batch:
        if isinstance(ele, torch.Tensor):
            new_batch.append(ele.to(device))
        else:
            new_batch.append(ele)
    return new_batch


def dict_to_device(batch, device='cuda'):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def to_list_tuples_coords(anns):
    new_anns = []
    for ann in anns:
        points = []
        for x, y in ann:
            points.append((x[0].tolist(), y[0].tolist()))
        new_anns.append(points)
    return new_anns


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))



def minmax_scaler_img(img):
    img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype(
        'uint8')  # noqa
    return img


def visualize_tfb(tfb_writer,
                  imgs,
                  preds,
                  global_steps,
                  thresh=0.5,
                  mode="TRAIN"):
    # origin img
    # imgs.shape = (batch_size, 3, image_size, image_size)
    imgs = torch.stack([
        torch.Tensor(
            minmax_scaler_img(img_.to('cpu').numpy().transpose((1, 2, 0))))
        for img_ in imgs
    ])
    
    imgs = torch.Tensor(imgs.numpy().transpose((0, 3, 1, 2)))
    imgs_grid = torch_utils.make_grid(imgs)
    imgs_grid = torch.unsqueeze(imgs_grid, 0)
    # imgs_grid.shape = (3, image_size, image_size * batch_size)
    tfb_writer.add_images('{}/origin_imgs'.format(mode), imgs_grid,
                          global_steps)

    # pred_prob_map / pred_thresh_map
    pred_prob_map = preds[:, 0, :, :]
    pred_thred_map = preds[:, 1, :, :]
    pred_prob_map[pred_prob_map <= thresh] = 0
    pred_prob_map[pred_prob_map > thresh] = 1

    # make grid
    pred_prob_map = pred_prob_map.unsqueeze(1)
    pred_thred_map = pred_thred_map.unsqueeze(1)

    probs_grid = torch_utils.make_grid(pred_prob_map, padding=0)
    probs_grid = torch.unsqueeze(probs_grid, 0)
    probs_grid = probs_grid.detach().to('cpu')

    thres_grid = torch_utils.make_grid(pred_thred_map, padding=0)
    thres_grid = torch.unsqueeze(thres_grid, 0)
    thres_grid = thres_grid.detach().to('cpu')

    tfb_writer.add_images('{}/prob_imgs'.format(mode), probs_grid,
                          global_steps)
    tfb_writer.add_images('{}/thres_imgs'.format(mode), thres_grid,
                          global_steps)





def draw_bbox(img, result, color=(255, 0, 0), thickness=3):
    """
    :input: RGB img
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    img = img.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img, [point], True, color, thickness)
    return img
