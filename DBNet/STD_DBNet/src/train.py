import os
import gc
import time
import random
import warnings

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as torch_optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from losses import DBLoss
from models import DBTextModel
from text_metrics import (cal_text_score, RunningScore, QuadMetric)
from utils import (setup_determinism, dict_to_device,
                   visualize_tfb, to_device)
from postprocess import SegDetectorRepresenter
from data_loaders import DatasetIter
warnings.filterwarnings('ignore')
cv2.setNumThreads(0)
def load_model(model_path, device):
    assert os.path.exists(model_path)
    dbnet = DBTextModel().to(device)
    dbnet.load_state_dict(torch.load(model_path,
                                     map_location=device))
    return dbnet

def print_vip(a):
    print(a)
    
    with open("/home/lab/khanhnd/STD_DBNet/saved_models/"+cfg["result_dir"]+"/history.txt", "a") as f:
        f.write(a+"\n")
        
def get_data_loaders(cfg):
    ignore_tags = cfg["data"]["vietnamese"]['ignore_tags']
    train_dir =cfg["data"]["vietnamese"]['train_dir']
    test_dir=cfg["data"]["vietnamese"]['test_dir']
    valid_dir = cfg["data"]["vietnamese"]['valid_dir']
    train_gt_dir = cfg["data"]["vietnamese"]['train_gt_dir']
    valid_gt_dir =cfg["data"]["vietnamese"]['valid_gt_dir']
    test_gt_dir =cfg["data"]["vietnamese"]['test_gt_dir']
    train_iter = DatasetIter(train_dir,
                                    train_gt_dir,
                                   ignore_tags,
                                    image_size=cfg["hps"]["img_size"],
                                    is_training=True,
                                    debug=False)
    valid_iter = DatasetIter(valid_dir,
                                valid_gt_dir,
                                ignore_tags,
                                image_size=cfg["hps"]["img_size"],
                                is_training=False,
                                debug=False)
    
    test_iter = DatasetIter(test_dir,
                                test_gt_dir,
                                ignore_tags,
                                image_size=cfg["hps"]["img_size"],
                                is_training=False,
                                debug=False)

    train_loader = DataLoader(dataset=train_iter,
                                batch_size=cfg["hps"]["batch_size"],
                                shuffle=True,
                                num_workers=1)
    valid_loader = DataLoader(dataset=valid_iter,
                                batch_size=cfg["hps"]["test_batch_size"],
                                shuffle=False,
                                num_workers=0)
    test_loader = DataLoader(dataset=test_iter,
                                batch_size=cfg["hps"]["test_batch_size"],
                                shuffle=False,
                                num_workers=0)
    
    return train_loader, valid_loader, test_loader


def main(cfg):

    # set determinism
    setup_determinism(42)


    # create result dir 
    if not os.path.exists("/home/lab/khanhnd/STD_DBNet/saved_models/"+cfg["result_dir"]):
        os.makedirs("/home/lab/khanhnd/STD_DBNet/saved_models/"+cfg["result_dir"])

    with open("/home/lab/khanhnd/STD_DBNet/saved_models/"+cfg["result_dir"]+"/history.txt", "w") as f:
        pass
    # setup log folder
    log_dir_path = os.path.join(cfg["meta"]["root_dir"], "logs")
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    tfb_log_dir = os.path.join(log_dir_path, str(cfg["result_dir"]))
    print_vip(tfb_log_dir)
    if not os.path.exists(tfb_log_dir):
        os.makedirs(tfb_log_dir)
    tfb_writer = SummaryWriter(tfb_log_dir)

    device = cfg["meta"]["device"]
    print_vip(device)
    model_path=cfg["load_path"]
    dbnet = DBTextModel().to(device)
    if cfg["continue"]==True:
            print("Loading previous checkpoint")
            dbnet=load_model(model_path,device)
            # print("Wrong checkpoint file path")
    else: 
        print("Start from scratch")    
        
    lr_optim = cfg["optimizer"]['lr']

    dbnet.train()
    cfg_optimizer=cfg["optimizer"]
    criterion = DBLoss(alpha=cfg_optimizer["alpha"],
                       beta=cfg_optimizer["beta"],
                       negative_ratio=cfg_optimizer["negative_ratio"],
                       reduction=cfg_optimizer["reduction"]).to(device)
    db_optimizer = torch_optim.Adam(dbnet.parameters(),
                                    lr=lr_optim)

    

    # get data loaders
    dataset_name = cfg["dataset"]["name"]
    train_loader, valid_loader, test_loader = get_data_loaders(cfg)

    # train model
    print_vip("Start training!")
    torch.cuda.empty_cache()
    gc.collect()
    global_steps = 0

    with open("/home/lab/khanhnd/STD_DBNet/saved_models/"+cfg["result_dir"]+"/test_result.txt","w") as f:
                    f.write("hmean,accuracy,precision,recall\n")


    # setup model checkpoint
    
    if cfg["continue"]==False:
        
        with open("/home/lab/khanhnd/STD_DBNet/saved_models/"+cfg["result_dir"]+"/valid_result.txt", "w") as f:
            f.write("best_hmean,best_train_loss,best_valid_loss\n")
            
            best_valid_loss = np.inf
            best_train_loss = np.inf
            best_hmean = 0
    else:
        with open("/home/lab/khanhnd/STD_DBNet/saved_models/"+cfg["result_dir"]+"/valid_result.txt", "r") as f:
            lst=f.readlines()
            res=lst[-1].strip().split(",")
            best_valid_loss = float(res[2])
            best_train_loss = float(res[1])
            best_hmean = float(res[0])


    best_hmean_epoch=0
    best_loss_epoch=0

    for epoch in range(cfg["hps"]["no_epochs"]):

        # TRAINING
        dbnet.train()
        train_loss = 0
        running_metric_text = RunningScore(cfg["hps"]["no_classes"])
        for batch_index, batch in enumerate(train_loader):
            lr = db_optimizer.param_groups[0]['lr']
            global_steps += 1

            batch = dict_to_device(batch, device=device)
            preds = dbnet(batch['img'])
            assert preds.size(1) == 3
            _batch = torch.stack([
                batch['prob_map'], batch['supervision_mask'],
                batch['thresh_map'], batch['text_area_map']
            ])
            prob_loss, threshold_loss, binary_loss, prob_threshold_loss, total_loss = criterion(  # noqa
                preds, _batch)
            db_optimizer.zero_grad()

            total_loss.backward()
            db_optimizer.step()

            score_shrink_map = cal_text_score(
                preds[:, 0, :, :],
                batch['prob_map'],
                batch['supervision_mask'],
                running_metric_text,
                thresh=cfg['metric']['thred_text_score'])

            train_loss += total_loss
            acc = score_shrink_map['Mean Acc']
            iou_shrink_map = score_shrink_map['Mean IoU']

            # tf-board
            tfb_writer.add_scalar('TRAIN/LOSS/total_loss', total_loss,
                                  global_steps)
            tfb_writer.add_scalar('TRAIN/LOSS/loss', prob_threshold_loss,
                                  global_steps)
            tfb_writer.add_scalar('TRAIN/LOSS/prob_loss', prob_loss,
                                  global_steps)
            tfb_writer.add_scalar('TRAIN/LOSS/threshold_loss', threshold_loss,
                                  global_steps)
            tfb_writer.add_scalar('TRAIN/LOSS/binary_loss', binary_loss,
                                  global_steps)
            tfb_writer.add_scalar('TRAIN/ACC_IOU/acc', acc, global_steps)
            tfb_writer.add_scalar('TRAIN/ACC_IOU/iou_shrink_map',
                                  iou_shrink_map, global_steps)
            tfb_writer.add_scalar('TRAIN/HPs/lr', lr, global_steps)

            if global_steps % cfg['hps']['log_iter'] == 0:
                print_vip(
                    "[{}-{}] - lr: {} - total_loss: {} - loss: {} - acc: {} - iou: {}"  # noqa
                    .format(epoch + 1, global_steps, lr, total_loss,
                            prob_threshold_loss, acc, iou_shrink_map))

        end_epoch_loss = train_loss / len(train_loader)
        print_vip("Train loss: {}".format(end_epoch_loss))
        gc.collect()
        
        # TFB IMGs
        # shuffle = True
        visualize_tfb(tfb_writer,
                      batch['img'],
                      preds,
                      global_steps=global_steps,
                      thresh=cfg['metric']['thred_text_score'],
                      mode="TRAIN")

        seg_obj = SegDetectorRepresenter(thresh=cfg['metric']['thred_text_score'],
                                         box_thresh=cfg['metric']['prob_threshold'],
                                         unclip_ratio=cfg['metric']['unclip_ratio'])
        metric_cls = QuadMetric()

        # EVAL
        dbnet.eval()
        valid_running_metric_text = RunningScore(cfg['hps']['no_classes'])
        test_running_metric_text = RunningScore(cfg['hps']['no_classes'])

        valid_loss = 0
        raw_metrics = []
        valid_visualize_index = random.choice(range(len(valid_loader)))
        for valid_batch_index, valid_batch in tqdm(
                enumerate(valid_loader),
                total=len(valid_loader)):

            with torch.no_grad():
                valid_batch = dict_to_device(valid_batch, device)

                valid_preds = dbnet(valid_batch['img'])
                assert valid_preds.size(1) == 2

                _batch = torch.stack([
                    valid_batch['prob_map'], valid_batch['supervision_mask'],
                    valid_batch['thresh_map'], valid_batch['text_area_map']
                ])
                valid_total_loss = criterion(valid_preds, _batch)
                valid_loss += valid_total_loss

                # visualize predicted image with tfb
                if valid_batch_index == valid_visualize_index:
                    
                    visualize_tfb(tfb_writer,
                                  valid_batch['img'],
                                  valid_preds,
                                  global_steps=global_steps,
                                  thresh=cfg["metric"]["thred_text_score"],
                                  mode="VALID")

                valid_score_shrink_map = cal_text_score(
                    valid_preds[:, 0, :, :],
                    valid_batch['prob_map'],
                    valid_batch['supervision_mask'],
                    valid_running_metric_text,
                    thresh=cfg["metric"]["thred_text_score"])
                valid_acc = valid_score_shrink_map['Mean Acc']
                valid_iou_shrink_map = valid_score_shrink_map['Mean IoU']
                tfb_writer.add_scalar('VALID/LOSS/val_loss', valid_total_loss,
                                      global_steps)
                tfb_writer.add_scalar('VALID/ACC_IOU/val_acc', valid_acc,
                                      global_steps)
                tfb_writer.add_scalar('VALID/ACC_IOU/val_iou_shrink_map',
                                      valid_iou_shrink_map, global_steps)

                # Cal P/R/Hmean
                batch_shape = {'shape': [(cfg["hps"]["img_size"], cfg["hps"]["img_size"])]}
                box_list, score_list = seg_obj(
                    batch_shape,
                    valid_preds,
                    is_output_polygon=cfg["metric"]["is_output_polygon"])
                raw_metric = metric_cls.validate_measure(
                    valid_batch, (box_list, score_list))
                raw_metrics.append(raw_metric)
        metrics = metric_cls.gather_measure(raw_metrics)
        recall = metrics['recall'].avg
        precision = metrics['precision'].avg
        hmean = metrics['fmeasure'].avg

        temp_best_hmean= best_hmean
        if hmean >= best_hmean:
            best_hmean_epoch=epoch
            print("SAVING BEST F1")
            
            best_hmean = hmean
            torch.save(
                dbnet.state_dict(),
                os.path.join(cfg["meta"]['root_dir'],"saved_models",cfg["result_dir"],cfg["model"]['best_hmean_cp_path']))
            
            best_hmean_epoch=epoch
            print("DONE")

        print_vip(
            "VALID/Recall: {} - VALID/Precision: {} - VALID/HMean: {}".format(
                recall, precision, hmean))
        tfb_writer.add_scalar('VALID/recall', recall, global_steps)
        tfb_writer.add_scalar('VALID/precision', precision, global_steps)
        tfb_writer.add_scalar('VALID/hmean', hmean, global_steps)

        valid_loss = valid_loss / len(valid_loader)
        print_vip("[{}] - valid_loss: {}".format(global_steps, valid_loss))

        if valid_loss <= best_valid_loss and train_loss <= best_train_loss:
            best_loss_epoch=epoch
            print("SAVING BETTER LOSS CHECKPOINT")
            best_valid_loss = valid_loss
            best_train_loss = train_loss
            torch.save(dbnet.state_dict(),
                       os.path.join(cfg['meta']['root_dir'],"saved_models",cfg["result_dir"],cfg['model']['best_cp_path'] ))
            best_loss_epoch=epoch

            print("DONE")
        with open("/home/lab/khanhnd/STD_DBNet/saved_models/"+cfg["result_dir"]+"/valid_result.txt", "a") as f:
            f.write(str(best_hmean_epoch)+"_"+str(best_hmean)+","+ str(best_loss_epoch)+"_"+str(float(best_train_loss.item()))+","+str(best_loss_epoch)+"_"+str(float(best_valid_loss.item()))+"\n")   
        torch.cuda.empty_cache()
        gc.collect()


        if hmean > temp_best_hmean:
            test_visualize_index = random.choice(range(len(test_loader)))
            for test_batch_index, test_batch in tqdm(
                enumerate(test_loader),
                total=len(test_loader)):

                with torch.no_grad():
                    test_batch = dict_to_device(test_batch, device)

                    test_preds = dbnet(test_batch['img'])
                    assert valid_preds.size(1) == 2

                    _batch = torch.stack([
                        test_batch['prob_map'], test_batch['supervision_mask'],
                        test_batch['thresh_map'], test_batch['text_area_map']
                    ])

                    # visualize predicted image with tfb
                    if test_batch_index == test_visualize_index:
                        visualize_tfb(tfb_writer,
                                    test_batch['img'],
                                    test_preds,
                                    global_steps=global_steps,
                                    thresh=cfg["metric"]["thred_text_score"],
                                    mode="TEST")

                    test_score_shrink_map = cal_text_score(
                        test_preds[:, 0, :, :],
                        test_batch['prob_map'],
                        test_batch['supervision_mask'],
                        test_running_metric_text,
                        thresh=cfg["metric"]["thred_text_score"])
                    test_acc = test_score_shrink_map['Mean Acc']

                    batch_shape = {'shape': [(cfg["hps"]["img_size"], cfg["hps"]["img_size"])]}
                    box_list, score_list = seg_obj(
                        batch_shape,
                        test_preds,
                        is_output_polygon=cfg["metric"]["is_output_polygon"])
                    
                    raw_metric = metric_cls.validate_measure(
                        test_batch, (box_list, score_list))
                    raw_metrics.append(raw_metric)
            metrics = metric_cls.gather_measure(raw_metrics)
            recall = metrics['recall'].avg
            precision = metrics['precision'].avg
            hmean = metrics['fmeasure'].avg
            with open("/home/lab/khanhnd/STD_DBNet/saved_models/"+cfg["result_dir"]+"/test_result.txt","a") as f:
                f.write(str(hmean)+","+str(test_acc)+","+str(precision)+","+str(recall)+"\n")


    print_vip("Training completed")
    print("SAVING LAST EPOCH")
    torch.save(dbnet.state_dict(),
               os.path.join(cfg['meta']['root_dir'],"saved_models" ,cfg["result_dir"],str(epoch)+"_"+cfg['model']['last_cp_path']))
    print_vip("Saved model")
    
        



path="/home/lab/khanhnd/STD_DBNet/config.yaml"
import yaml
from yaml.loader import SafeLoader

# Open the file and load the file
with open(path) as f:
    cfg = yaml.load(f, Loader=SafeLoader)


def run():
    main(cfg)


if __name__ == '__main__':
    run()
