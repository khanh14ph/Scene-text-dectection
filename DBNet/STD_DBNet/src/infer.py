import cv2
import numpy as np
from models import DBTextModel
import os
import gc
import torch
from postprocess import SegDetectorRepresenter
model_path="/home/lab/khanhnd/STD_DBNet/saved_models/result_200_0.0003_advanced_aug/best_hmean_cp.pth"
device="cpu"

def read_img(img_fp):
    img = cv2.imread(img_fp)[:, :, ::-1]
    h_origin, w_origin, _ = img.shape
    return img, h_origin, w_origin
def test_resize(img,pad, size=640):
    h, w, c = img.shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)

    new_img = None
    if pad:
        mean=np.mean(img, axis=(0,1))
    
        padimg = np.zeros((size, size, c), img.dtype)
        padimg[:,:,0]=np.full((size, size), mean[0])
        padimg[:,:,1]=np.full((size,size), mean[1])
        padimg[:,:,2]=np.full((size,size), mean[2])
        padimg[:h, :w] = cv2.resize(img, (w, h))
        new_img=padimg
    else:
        new_img = cv2.resize(img, (w, h))

    return new_img
def test_preprocess(img,
                    mean=[103.939, 116.779, 123.68],
                    to_tensor=True,
                    pad=True):
    img = test_resize(img,pad, size=640)

    img = img.astype(np.float32)
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]
    img = np.expand_dims(img, axis=0)

    if to_tensor:
        img = torch.Tensor(img.transpose(0, 3, 1, 2))

    return img
def minmax_scaler_img(img):
    img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype(
        'uint8')  # noqa
    return img
def visualize_heatmap(thred, tmp_img, tmp_pred, save_path):
    pred_prob = tmp_pred[0]
    t=tmp_pred[1]
    pred_prob[pred_prob <= thred] = 0
    pred_prob[pred_prob > thred] = 1

    np_img = minmax_scaler_img(tmp_img[0].to(device).numpy().transpose(
        (1, 2, 0)))
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    # plt.imsave(save_path, np_img)
    # cv2.imwrite(save_path, np_img)
    plt.imshow(np_img)
    plt.imshow(pred_prob, cmap='jet', alpha=0.6)
    plt.savefig(save_path,dpi=200,bbox_inches='tight')
    gc.collect()

import imageio
import matplotlib.pyplot as plt
def draw_bbox(img, result, color=(0, 255, 0), thickness=3):
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
args=dict()
args["thresh"]=0.5
args['box_thresh']=0.7
args['unclip_ratio']=1.5
args['is_output_polygon']=False
args['alpha']=0.6
path="/home/lab/khanhnd/STD_DBNet/config.yaml"
import yaml
from yaml.loader import SafeLoader

# Open the file and load the file
with open(path) as f:
    cfg = yaml.load(f, Loader=SafeLoader)
def visualize_polygon(cfg, img_path, preds,  save_path):
    img=cv2.imread(img_path)
    img=test_resize(img,pad=True)
    cfg["metric"]["is_output_polygon"]=False
    img_origin, h_origin, w_origin = img, 640,640
    batch = {'shape': [(h_origin, w_origin)]}
    seg_obj=SegDetectorRepresenter(thresh=cfg['metric']['thred_text_score'],
                                         box_thresh=cfg['metric']['prob_threshold'],
                                         unclip_ratio=cfg['metric']['unclip_ratio'])
    box_list, score_list = seg_obj(batch,
                                   preds,
                                   cfg["metric"]["is_output_polygon"])
    print(cfg["metric"]["is_output_polygon"])
    box_list, score_list = box_list[0], score_list[0]

    if len(box_list) > 0:
        if cfg["metric"]["is_output_polygon"]:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0
            box_list, score_list = box_list[idx], score_list[idx]
    else:
        box_list, score_list = [], []
    tmp_img = draw_bbox(img, np.array(box_list))
    tmp_pred = cv2.resize(preds[0, 0, :, :].cpu().numpy(),
                          (w_origin, h_origin))

    # plt.imsave(save_path, tmp_pred)
    # cv2.imwrite(save_path, tmp_pred)
    plt.imshow(tmp_img)
    plt.savefig(save_path,
                dpi=200,
                bbox_inches='tight')
    gc.collect()
def load_model(model_path):
    assert os.path.exists(model_path)
    dbnet = DBTextModel().to(device)
    dbnet.load_state_dict(torch.load(model_path,
                                     map_location=device))
    return dbnet
net=load_model(model_path)
import glob
test_lst= glob.glob("/home/lab/khanhnd/STD_DBNet/dataset/vietnamese/test_images/*", recursive=True)
total_mean=[115.31973217, 120.95974333, 125.89291732]
test_lst.sort()
threshold=0.5
import tqdm
import time
for i in tqdm.tqdm(test_lst):
    start = time.time()
    heat_save_path="/home/lab/khanhnd/STD_DBNet/inference_test/heat_map/"+i.split("/")[-1][:-4]+"_heat.jpg"
    polygon_save_path="/home/lab/khanhnd/STD_DBNet/inference_test/polygon/"+i.split("/")[-1][:-4]+"_polygon.jpg"
    img=cv2.imread(i)[:,:,::-1]
    tmp_img=test_preprocess(img, total_mean, to_tensor=True,
                              pad=True).to(device)
    with torch.no_grad():
        preds = net(tmp_img)
        visualize_heatmap(threshold,tmp_img, preds.to('cpu')[0].numpy(), heat_save_path )   
        visualize_polygon(cfg,i, preds, polygon_save_path)
    print(">>> Inference took {}'s".format(time.time() - start))