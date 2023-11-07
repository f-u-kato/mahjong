from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation


from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
import torch.backends.cudnn as cudnn
import src.out_func.draw_img as draw
import src.get_func.get_img as get
import src.eval.eval as eval
from PIL import Image

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    new_image = Image.fromarray(new_image)
    return new_image

# TRAINED_MODEL=r'weights\yolact_mahjongCP_854_400000.pth'
# TRAINED_MODEL=r'weights\yolact_mahjongCP_2_120.pth'
TRAINED_MODEL=r'weights\yolact_mahjong_12499_400000.pth'
# TRAINED_MODEL=r"weights\yolact_mahjongCP_13_6084.pth"
TRAINED_MODEL=r"weights\yolact_mahjong_hand_11_2062.pth"
TRAINED_MODEL=r"weights\yolact_mahjong_hand_39_7301.pth"
TRAINED_MODEL=r"weights\yolact_mahjong_hand_6_1887.pth"
TRAINED_MODEL=r"weights\yolact_mahjong_naki_11_2071.pth"
TRAINED_MODEL=r"weights\yolact_mahjong_create_31_11830.pth"
TRAINED_MODEL=r"weights\yolact_mahjong_create_53_19939.pth"
TRAINED_MODEL=r"weights\yolact_mahjong_create_79_30000.pth"
TRAINED_MODEL=r"weights\yolact_mahjong_create_106_40000.pth"
TRAINED_MODEL=r"weights\yolact_mahjong_create_133_50000.pth"
# TRAINED_MODEL=r"weights\yolact_mahjong_create_159_60000.pth"
OLD_TRAINED_MODEL=r"weights\yolact_mahjong_create_186_70000.pth"
# TRAINED_MODEL=r"weights\yolact_mahjong_create_213_80000.pth"
# TRAINED_MODEL=r"weights\yolact_mahjong_create_239_90000.pth"
# TRAINED_MODEL=r"weights\new_create\yolact_mahjong_create_293_110000.pth"
# TRAINED_MODEL=r"weights\new_create\yolact_mahjong_create_186_70000.pth"
# TRAINED_MODEL=r"weights\810\yolact_mahjong_create_239_90000.pth"
TRAINED_MODEL=r"weights\810\yolact_mahjong_create_213_80000.pth"
# TRAINED_MODEL=r"weights\810\yolact_mahjong_create_186_70000.pth"
TRAINED_MODEL=r"weights\new\yolact_mahjong_create_79_50000.pth"
TRAINED_MODEL=r"weights\new\yolact_mahjong_create_63_40000.pth"
TRAINED_MODEL=r"weights\new\yolact_mahjong_create_15_10000.pth"
HAND_TRAINED_MODEL=r"weights\new\yolact_mahjong_create_31_20000.pth"
HAND_TRAINED_MODEL=r"weights\yolact_mahjong_hand_32_10000.pth"
DORA_TRAINED_MODEL=r"weights\yolact_mahjong_dora_64_20000.pth"
NAKI_TRAINED_MODEL=r"weights\yolact_mahjong_naki_32_10000.pth"
# TRAINED_MODEL=r"weights\new\yolact_mahjong_create_47_30000.pth"
# TRAINED_MODEL=r"weights\dora_plus\yolact_mahjong_create_31_20000.pth"
# TRAINED_MODEL=r"weights\dora_plus\yolact_mahjong_create_47_30000.pth"
# TRAINED_MODEL=r"weights\dora_plus\yolact_mahjong_create_63_40000.pth"
TRAINED_MODEL=r"weights\hand_naki_dora_win\yolact_mahjong_create_47_30000.pth"
ALL_TRAINED_MODEL=r"weights\hand_naki_dora_win\yolact_mahjong_create_31_20000.pth"
# TRAINED_MODEL=r"weights\hand_naki_dora_win\yolact_mahjong_create_15_10000.pth"



image_size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

def trigger_eval(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img=cv2pil(img)
    model =torch.load('./weights/resnet18/model_20.pth')
    model=model.to(device)
    input = transform(img).unsqueeze(dim=0).to(device)
    outputs = model(input)
    class_id = int(outputs.argmax(dim=1)[0])
    return class_id

def riichi_eval(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img=cv2pil(img)
    model =torch.load('./weights/riichi/model_20.pth')
    model=model.to(device)
    input = transform(img).unsqueeze(dim=0).to(device)
    outputs = model(input)
    class_id = int(outputs.argmax(dim=1)[0])
    return class_id


def iou_np(a, b, a_area, b_area):
    abx_mn = np.maximum(a[0], b[:,0]) # xmin
    aby_mn = np.maximum(a[1], b[:,1]) # ymin
    abx_mx = np.minimum(a[2], b[:,2]) # xmax
    aby_mx = np.minimum(a[3], b[:,3]) # ymax
    # 共通部分の幅を計算。共通部分が無ければ0
    w = np.maximum(0, abx_mx - abx_mn + 1)
    # 共通部分の高さを計算。共通部分が無ければ0
    h = np.maximum(0, aby_mx - aby_mn + 1)
    # 共通部分の面積を計算。共通部分が無ければ0
    intersect = w*h
    
    # N個のbについて、aとのIoUを一気に計算
    iou_np = intersect / (a_area + b_area - intersect)
    return iou_np
def nms_fast(bboxes, scores, classes,masks, iou_threshold=0.5):
    areas = (bboxes[:,2] - bboxes[:,0] + 1) \
             * (bboxes[:,3] - bboxes[:,1] + 1)
    sort_index = np.argsort(scores)
    
    i = -1 # 未処理の矩形のindex
    while(len(sort_index) >= 2 - i):
        # score最大のindexを取得
        max_scr_ind = sort_index[i]
        # score最大以外のindexを取得
        ind_list = sort_index[:i]
        # score最大の矩形それ以外の矩形のIoUを計算
        iou = iou_np(bboxes[max_scr_ind], bboxes[ind_list], \
                     areas[max_scr_ind], areas[ind_list])
        
        # IoUが閾値iou_threshold以上の矩形を計算
        del_index = np.where(iou >= iou_threshold)
        # IoUが閾値iou_threshold以上の矩形を削除
        sort_index = np.delete(sort_index, del_index)
        #print(len(sort_index), i, flush=True)
        i -= 1 # 未処理の矩形のindexを1減らす
    
    # bboxes, scores, classesから削除されなかった矩形のindexのみを抽出
    bboxes = bboxes[sort_index]
    scores = scores[sort_index]
    classes = classes[sort_index]
    masks=masks[sort_index]
    
    return [classes, scores, bboxes,masks]

def delete_large_box(classes,scores,boxes,top_k):
    if len(classes)==0:
        return classes,scores,boxes
    best_box=boxes[0]
    new_classes=[classes[0]]
    new_scores=[scores[0]]
    new_boxes=[boxes[0]]
    best_area=(best_box[2]-best_box[0])*(best_box[3]-best_box[1])
    for i in range(1,top_k):
        if len(classes)<=i:
            break
        inter_area=multi_bbox_intersection_area(new_boxes,boxes[i],new_classes,classes[i])

        area=(boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1])
        if inter_area<area*0.6:
            new_classes.append(classes[i])
            new_scores.append(scores[i])
            new_boxes.append(boxes[i])
    for i in range(top_k,len(classes)):
        if len(new_classes)>=top_k:
            break
        print(i)
        inter_area=multi_bbox_intersection_area(new_boxes,boxes[i],new_classes,classes[i])
        area=(boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1])
        if inter_area<area*0.6:
            new_classes.append(classes[i])
            new_scores.append(scores[i])
            new_boxes.append(boxes[i])
        
    return new_classes,new_scores,new_boxes
def bbox_intersection_area(bbox1, bbox2):
    # 2つのバウンディングボックスの座標を取得する
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    # 重なっている部分の左上座標を求める
    x_left = max(x1, x3)
    y_top = max(y1, y3)

    # 重なっている部分の右下座標を求める
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    # 重なっている部分の幅と高さを計算する
    intersection_w = max(0, x_right - x_left )
    intersection_h = max(0, y_bottom - y_top )

    # 面積を計算する
    intersection_area = intersection_w * intersection_h

    return intersection_area

def multi_bbox_intersection_area(bboxes1, bbox2,classes1,class2):
    # 複数のバウンディングボックスと一つのバウンディングボックスの左上座標と右下座標を取得する
    interarea=0
    for box,class1 in zip(bboxes1,classes1):
        # if class1==class2:
            interarea+=bbox_intersection_area(box,bbox2)

    return interarea

def win_results(preds,w,h,score_threshold=0.5,top_k=1):
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
    return [classes,scores,boxes]

def multi_win_eval(imgs,score_threshold=0.5):
    images=[draw.padding_img_size(img.copy()) for img in imgs]
    h, w, _ = images[0].shape
    
    model_path = SavePath.from_str(TRAINED_MODEL)
    config = model_path.model_name + '_config'
    set_cfg(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results=[]
    with torch.no_grad():
        net = Yolact()
        net.load_weights(TRAINED_MODEL)
        net.eval()
        net = net.to(device)
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False
        frames = [torch.from_numpy(img).to(device).float() for img in images]
        batch = torch.stack(frames)
        batch = FastBaseTransform()(batch)
        preds = net(batch)
        top_k=1
        for pred in preds:
            results.append(win_results(pred,w,h,score_threshold,top_k))

    return results

def win_eval(img,score_threshold=0.5):
    img=draw.padding_img_size(img.copy())
    h, w, _ = img.shape
    
    model_path = SavePath.from_str(TRAINED_MODEL)
    # TODO: Bad practice? Probably want to do a name lookup instead.
    config = model_path.model_name + '_config'
    set_cfg(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        net = Yolact()
        net.load_weights(TRAINED_MODEL)
        net.eval()
        net = net.to(device)
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False
        frame = torch.from_numpy(img).to(device).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = net(batch)
        top_k=1
        [classes,scores,boxes]=win_results(preds,w,h,score_threshold,top_k)
    
    return classes,scores,boxes

def dora_results(preds,w,h,score_threshold=0.5,top_k=8):
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

    return [classes,scores,boxes]

def dora_eval(img,score_threshold=0.5):
    img=draw.padding_img(img.copy())
    h, w, _ = img.shape
    
    model_path = SavePath.from_str(TRAINED_MODEL)
    # TODO: Bad practice? Probably want to do a name lookup instead.
    config = model_path.model_name + '_config'
    set_cfg(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        net = Yolact()
        net.load_weights(DORA_TRAINED_MODEL)
        net.eval()
        net = net.to(device)
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False
        frame = torch.from_numpy(img).to(device).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = net(batch)
        top_k=8
        [classes,scores,boxes]=dora_results(preds,w,h,score_threshold,top_k)
    return classes,scores,boxes

def hand_results(preds,w,h,score_threshold=0.5,top_k=13):
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(preds, w, h, score_threshold = score_threshold)
        cfg.rescore_bbox = save
    
    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)
        
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        classes, scores, boxes = delete_large_box(classes,scores,boxes,top_k)
    return [classes,scores,boxes]

def hand_eval(img,score_threshold=0.5):
    img=draw.padding_img(img.copy())
    h, w, _ = img.shape
    
    model_path = SavePath.from_str(TRAINED_MODEL)
    # TODO: Bad practice? Probably want to do a name lookup instead.
    config = model_path.model_name + '_config'
    set_cfg(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        net = Yolact()
        net.load_weights(HAND_TRAINED_MODEL)
        net.eval()
        net = net.to(device)
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = True
        cfg.mask_proto_debug = False
        frame = torch.from_numpy(img).to(device).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = net(batch)
        top_k=13
        [classes,scores,boxes]=hand_results(preds,w,h,score_threshold,top_k)

    return classes,scores,boxes


def naki_results(preds,w,h,score_threshold=0.5,top_k=16):
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

    return [classes,scores,boxes]

def naki_eval(img,score_threshold=0.5):
    img=draw.padding_img(img)
    h, w, _ = img.shape
    
    model_path = SavePath.from_str(OLD_TRAINED_MODEL)
    # TODO: Bad practice? Probably want to do a name lookup instead.
    config = model_path.model_name + '_config'
    set_cfg(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        net = Yolact()
        net.load_weights(NAKI_TRAINED_MODEL)
        net.eval()
        net = net.to(device)
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False
        frame = torch.from_numpy(img).to(device).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = net(batch)
        top_k=16
        [classes,scores,boxes]=naki_results(preds,w,h,score_threshold,top_k)
    return classes,scores,boxes

def without_win_eval(imgs,score_threshold=0.5):
    images=[draw.padding_img_size(img.copy()) for img in imgs]
    hand_h,hand_w,_=images[0].shape
    dora_h,dora_w,_=images[1].shape
    naki_h,naki_w,_=images[2].shape
    
    model_path = SavePath.from_str(ALL_TRAINED_MODEL)
    config = model_path.model_name + '_config'
    set_cfg(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    with torch.no_grad():
        net = Yolact()
        net.load_weights(TRAINED_MODEL)
        net.eval()
        net = net.to(device)
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False
        frames = [torch.from_numpy(img).to(device).float() for img in images]
        batch = torch.stack(frames)
        batch = FastBaseTransform()(batch)
        preds = net(batch)
        
        [hand_classes,hand_scores,hand_boxes]=hand_results(preds[0],hand_w,hand_h,score_threshold,13)
        [dora_classes,dora_scores,dora_boxes]=dora_results(preds[1],dora_w,dora_h,score_threshold,8)
        [naki_classes,naki_scores,naki_boxes]=naki_results(preds[2],naki_w,naki_h,score_threshold,16)
    return [hand_classes,hand_scores,hand_boxes],[dora_classes,dora_scores,dora_boxes],[naki_classes,naki_scores,naki_boxes]
    
import cv2
MAHJONG_CLASSES = ("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                   "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                   "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
                   "ton", "nan", "sha", "pe",
                   "haku", "hatsu", "chun",
                   "aka_5m", "aka_5p", "aka_5s",
                   "ura")

def prep_display(classes,boxes,img):
    color=[(255,0,0),(255,255,0),(255,0,255),(0,255,255),(0,0,255)]
    count=0
    for num , box in zip(classes,boxes):
        if count==len(color):
            count=0
        [x1,y1,x2,y2]=box
        img=cv2.rectangle(img, (x1,y1), (x2,y2), color[count],3)
        img=cv2.putText(img,MAHJONG_CLASSES[num],(x1,y1),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,color=color[count])
        count+=1
    return img


    




if __name__=='__main__':
        i=33
        for i in range(22):
            print(i)
            if i == 8:
                continue
            img=cv2.imread(f'./data/test/hand{i}.png')
            c,s,b=hand_eval(img)
            img_numpy = prep_display(c,b,img)
            
            cv2.imshow('',img_numpy)
            cv2.waitKey()
            continue
            
            c.sort()
            for num in c:
                print(MAHJONG_CLASSES[num],end=',')
            print('')
