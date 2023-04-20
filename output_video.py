import cv2
import math
from data import COCODetection, get_label_map, MEANS, COLORS
from src.yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

def angle(pt1, pt2, pt0) -> float:
    dx1 = float(pt1[0, 0] - pt0[0, 0])
    dy1 = float(pt1[0, 1] - pt0[0, 1])
    dx2 = float(pt2[0, 0] - pt0[0, 0])
    dy2 = float(pt2[0, 1] - pt0[0, 1])
    v = math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2))
    return (dx1*dx2 + dy1*dy2) / v

def padding_img(img):
    color = (130,180,0)
    #16:9
    h,w,_=img.shape
    if w*9//16 < h:
        h_pad=0
        w_pad=h*16//9-w
    else:
        h_pad=w*9//16-h
        w_pad=0
    return cv2.copyMakeBorder(img, h_pad, 0, w_pad, 0, cv2.BORDER_CONSTANT, value=color)

# 
def get_rect(path):
    return

#緑領域抽出
def get_green(path):
    if type(path) is str:
        img = cv2.imread(path)
    else:
        img=path.copy()
    height, width, channels = img.shape[:3]
    hsvLower = np.array([30, 50, 40])    # 抽出する色の下限(HSV)
    hsvUpper = np.array([90, 255, 255])    # 抽出する色の上限(HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 画像をHSVに変換
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSVからマスクを作成
    kernel = np.ones((3, 3), np.uint8)
    hsv_mask = cv2.morphologyEx(
        hsv_mask, cv2.MORPH_OPEN, kernel, iterations=3)  # クロージング
    kernel = np.ones((3, 3), np.uint8)
    hsv_mask = cv2.morphologyEx(
        hsv_mask, cv2.MORPH_CLOSE, kernel, iterations=20)  # クロージング
    # 輪郭取得
    contours, _ = cv2.findContours(
        hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    [h,w]=hsv_mask.shape
    min_area=h//3*w//3
    cut_points=[] 
    for i, cnt in enumerate(contours):
        arclen = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, arclen*0.1, True)
        area = abs(cv2.contourArea(approx))
        if approx.shape[0] == 4 and area > min_area  and cv2.isContourConvex(approx):
            maxCosine = 0
            for j in range(2, 5):
                cosine = abs(angle(approx[j % 4], approx[j-2], approx[j-1]))
                maxCosine = max(maxCosine, cosine)
            if maxCosine < 0.5:
                rcnt = approx.reshape(-1, 2)
                point=([width,height],[0,0])
                for i in rcnt:
                    if i[0] < point[0][0]:
                        point[0][0]=i[0]
                    if i[0] > point[1][0]:
                        point[1][0]=i[0]
                    if i[1] < point[0][1]:
                        point[0][1]=i[1]
                    if i[1] > point[1][1]:
                        point[1][1]=i[1]
                cut_points=point
    return cut_points

#size=(h,w)
#win_tile
def draw_rect(points,size=(2160,3840,3),img=None):
    if img is None:
        img=np.zeros(size,np.uint8)

    hai_size=max(points[1][0]-points[0][0],points[1][1]-points[0][1])//20
    hai_point=[hai_size*3,hai_size*2]
    color = (0, 0, 255)
    pt1 = [points[0][0]+hai_point[0],points[0][1]+hai_point[1]]
    pt2 = [x + hai_size for x in pt1] 
    cv2.rectangle(img, pt1, pt2, color,3)
    pt1 = [points[1][0]-hai_point[0],points[1][1]-hai_point[1]]
    pt2 = [x - hai_size for x in pt1] 
    cv2.rectangle(img, pt1, pt2, color,3)
    pt1 = [points[1][0]-hai_point[0],points[0][1]+hai_point[1]]
    pt2 = [pt1[0]-hai_size,pt1[1]+hai_size]
    cv2.rectangle(img, pt1, pt2, color,3)
    pt1 = [points[0][0]+hai_point[0],points[1][1]-hai_point[1]]
    pt2 = [pt1[0]+hai_size,pt1[1]-hai_size]
    cv2.rectangle(img, pt1, pt2, color,3)
    return img

def win_tile_cut(img,points):
    hai_size=max(points[1][0]-points[0][0],points[1][1]-points[0][1])//14
    hai_point=[hai_size*3,hai_size*2]
    
    pt1 = [points[0][0]+hai_point[0],points[0][1]+hai_point[1]]
    pt2 = [x + hai_size for x in pt1] 
    return img[pt1[0]:pt2[0],pt1[1]:pt2[1],:]



def show_img(img):

    # ウィンドウを作成する
    cv2.namedWindow("Projector Output", cv2.WINDOW_NORMAL)

    # 画像を表示する
    cv2.imshow("Projector Output", img)

    # プロジェクターに接続する
    cv2.setWindowProperty("Projector Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



def main():
    cap = cv2.VideoCapture(0)
    # カメラ調節
    while(cap.isOpened()):
        ret, im = cap.read()
        field_points=get_green(im)
        if len(field_points) > 0:
            img=draw_rect(field_points,img=im)
        cv2.imshow("Camera", im)
        # show_img(img)````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

    #-卓領域検出
    while(cap.isOpened()):
        ret, im = cap.read()
        field_points=get_green(im)
        if len(field_points) > 0:
            img=draw_rect(field_points,img=im)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
    print('get_green : ok')
    cv2.imshow("Camera", img)
    c=cv2.waitKey()
    while(cap.isOpened()):
        ret, im = cap.read()
        cv2.imshow("Camera", rect_cut(im,field_points))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    


    # video.release()

def win_eval(img):
    img=padding_img(img)
    h, w, _ = img.shape
    trained_model=r'weights\yolact_mahjongCP_854_400000.pth'
    model_path = SavePath.from_str(trained_model)
    # TODO: Bad practice? Probably want to do a name lookup instead.
    config = model_path.model_name + '_config'
    set_cfg(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    with torch.no_grad():
        net = Yolact()
        net.load_weights(trained_model)
        net.eval()
        net = net.to(device)
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False
        frame = torch.from_numpy(img).to(device).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = net(batch)
        
        score_threshold=0.6
        top_k=1
        
        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(preds, w, h, score_threshold = score_threshold)
            cfg.rescore_bbox = save

        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:top_k]
            
            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
    print(classes)

def hai_eval(img):
    img=padding_img(img)
    h, w, _ = img.shape
    trained_model=r'weights\yolact_mahjongCP_854_400000.pth'
    model_path = SavePath.from_str(trained_model)
    # TODO: Bad practice? Probably want to do a name lookup instead.
    config = model_path.model_name + '_config'
    set_cfg(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    with torch.no_grad():
        net = Yolact()
        net.load_weights(trained_model)
        net.eval()
        net = net.to(device)
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False
        frame = torch.from_numpy(img).to(device).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = net(batch)
        
        score_threshold=0.6
        top_k=13
        
        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(preds, w, h, score_threshold = score_threshold)
            cfg.rescore_bbox = save

        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:top_k]
            
            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
    print(classes)

def naki_eval(img):
    img=padding_img(img)
    h, w, _ = img.shape
    trained_model=r'weights\yolact_mahjongCP_854_400000.pth'
    model_path = SavePath.from_str(trained_model)
    # TODO: Bad practice? Probably want to do a name lookup instead.
    config = model_path.model_name + '_config'
    set_cfg(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    with torch.no_grad():
        net = Yolact()
        net.load_weights(trained_model)
        net.eval()
        net = net.to(device)
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False
        frame = torch.from_numpy(img).to(device).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = net(batch)
        
        score_threshold=0.6
        top_k=13
        
        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(preds, w, h, score_threshold = score_threshold)
            cfg.rescore_bbox = save

        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:top_k]
            
            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
    print(classes)

if __name__ == '__main__':
    path=r"F:\PBL\yolact\data\mahjong\sample\hai.png"
    img=cv2.imread(path)
    
    hai_eval(img)

    # main()
