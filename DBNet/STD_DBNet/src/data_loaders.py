import os
import glob
import math


import cv2
import numpy as np
from shapely.geometry import Polygon
import torch
from torch.utils.data import Dataset, DataLoader
import imgaug.augmenters as iaa
import pyclipper

from db_transforms import *


import imgaug.augmenters as iaa
class DatasetIter(Dataset):
    def __init__(self,
                 train_dir,
                 train_gt_dir,
                 ignore_tags,
                 is_training=True,
                 image_size=640,
                 min_text_size=8,
                 shrink_ratio=0.4,
                 thresh_min=0.3,
                 thresh_max=0.7,
                 augment=None,
                 mean=[103.939, 116.779, 123.68],
                 eval=False,
                 debug=False):

        self.train_dir = train_dir
        self.train_gt_dir = train_gt_dir
        self.ignore_tags = ignore_tags

        self.is_training = is_training
        self.image_size = image_size
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.augment = augment
        self.eval=eval
        if self.augment is None:
            self.augment = self._get_default_augment()

        self.mean = mean
        self.debug = debug

        # load metadata
        self.image_lst, self.gt_lst = self.load_metadata(
            train_dir, train_gt_dir)

        # load annotation
        self.all_anns = self.load_all_anns(self.gt_lst)
        assert len(self.image_lst) == len(self.all_anns)
    def load_metadata(self, img_dir, gt_dir):

        img_fps = glob.glob(os.path.join(img_dir, "*"))
        gt_fps = []
        for img_fp in img_fps:
            temp = img_fp.split("/")[-1]
            img_id=temp.split(".")[0].split("_")[1]
            gt_fn = temp+".txt"
            gt_fp = os.path.join(gt_dir, gt_fn)
            assert os.path.exists(img_fp)
            gt_fps.append(gt_fp)
        assert len(img_fps) == len(gt_fps)
        return img_fps, gt_fps
    def load_all_anns(self, gt_fps):
        res = []
        for gt_fp in gt_fps:
            lines = []
            with open(gt_fp, 'r',encoding='utf-8-sig') as f:
                for line in f:
                    item = {}
                    gt = line.strip().strip('\ufeff').strip(
                        '\xef\xbb\xbf').split(",",8)
                    label = ",".join(gt[8:])
                    poly = list(map(int, gt[:8]))
                    poly = np.asarray(poly).reshape(-1, 2).tolist()
                    item['poly'] = poly
                    item['text'] = label
                    lines.append(item)
            res.append(lines)
        return res

    def _get_default_augment(self):
        augment_seq = iaa.Sequential([
            iaa.Fliplr(0.2),
            iaa.AdditiveGaussianNoise(scale=(0, 0.14*255),per_channel=True), 
            iaa.Affine(rotate=(-30, 30),scale=(0.75, 1.2))
            # iaa.Resize((0.5, 2.0))
        ])
        return augment_seq

    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, index):

        image_path = self.image_lst[index]
        anns = self.all_anns[index]

        if self.debug:
            print(image_path)
            print(len(anns))

        img = cv2.imread(image_path)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.is_training and self.augment is not None:

            augment_seq = self.augment.to_deterministic()
            img, anns = transform(augment_seq, img, anns)

            img, anns = crop(img, anns)
            

        img, anns = resize(self.image_size, img, anns)

        anns = [ann for ann in anns if Polygon(ann['poly']).buffer(0).is_valid]
        gt = np.zeros((self.image_size, self.image_size),
                      dtype=np.float32)  # batch_gts
        mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
        thresh_map = np.zeros((self.image_size, self.image_size),
                              dtype=np.float32)  # batch_thresh_maps
        # batch_thresh_masks
        thresh_mask = np.zeros((self.image_size, self.image_size),
                               dtype=np.float32)

        if self.debug:
            print(type(anns), len(anns))

        ignore_tags = []
        for ann in anns:
            # i.e shape = (4, 2) / (6, 2) / ...
            poly = np.array(ann['poly'])
            height = max(poly[:, 1]) - min(poly[:, 1])
            width = max(poly[:, 0]) - min(poly[:, 0])
            polygon = Polygon(poly)

            # generate gt and mask
            if polygon.area < 1 or \
                    min(height, width) < self.min_text_size or \
                    ann['text'] in self.ignore_tags:
                ignore_tags.append(True)
                cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                continue
            else:
                # 6th equation
                distance = polygon.area * \
                    (1 - np.power(self.shrink_ratio, 2)) / polygon.length
        
                subject = [tuple(_l) for _l in ann['poly']]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)

                if len(shrinked) == 0:
                    ignore_tags.append(True)
                    cv2.fillPoly(mask,
                                 poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                else:
                    shrinked = np.array(shrinked[0]).reshape(-1, 2)
                    if shrinked.shape[0] > 2 and \
                            Polygon(shrinked).buffer(0).is_valid:
                        ignore_tags.append(False)
                        cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                    else:
                        ignore_tags.append(True)
                        cv2.fillPoly(mask,
                                     poly.astype(np.int32)[np.newaxis, :, :],
                                     0)
                        continue

            # generate thresh map and thresh mask

            thresh_map, thresh_mask=draw_thresh_map(ann['poly'],
                                          thresh_map,
                                          thresh_mask,
                                          shrink_ratio=self.shrink_ratio)

        thresh_map = thresh_map * \
            (self.thresh_max - self.thresh_min) + self.thresh_min
        
        # cv2.plt.savefig("/home/lab/khanhnd/STD_DBNet/1.jpg", img)
        img = img.astype(np.float32)
        img[..., 0] -= self.mean[0]
        img[..., 1] -= self.mean[1]
        img[..., 2] -= self.mean[2]

        img = np.transpose(img, (2, 0, 1))
        data_return = {
            "image_path": image_path,
            "img": img,
            "prob_map": gt,
            "supervision_mask": mask,
            "thresh_map": thresh_map,
            "text_area_map": thresh_mask,
        }
        if not self.is_training:
            data_return["anns"] = [ann['poly'] for ann in anns]
            data_return["ignore_tags"] = ignore_tags

        # return image_path, img, gt, mask, thresh_map, thresh_mask
        return data_return
