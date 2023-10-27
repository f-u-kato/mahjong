import math
import numpy as np
import cv2

def angle(pt1, pt2, pt0) -> float:
    dx1 = float(pt1[0, 0] - pt0[0, 0])
    dy1 = float(pt1[0, 1] - pt0[0, 1])
    dx2 = float(pt2[0, 0] - pt0[0, 0])
    dy2 = float(pt2[0, 1] - pt0[0, 1])
    v = math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2))
    return (dx1*dx2 + dy1*dy2) / v


#
def get_rect(path,source):
    image1=path.copy()
    image2=source.copy()
    image1=image1[:,:,0]
    image2=image2[:,:,0]
    diff = cv2.absdiff(image1, image2)
    threshold = 40
    _, hsv_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow("mask_old",cv2.resize(hsv_mask,[1920,1080]))
    kernel = np.ones((3, 3), np.uint8)
    hsv_mask = cv2.morphologyEx(
        hsv_mask, cv2.MORPH_OPEN, kernel, iterations=1)  # クロージング
    kernel = np.ones((3, 3), np.uint8)
    hsv_mask = cv2.morphologyEx(
        hsv_mask, cv2.MORPH_CLOSE, kernel, iterations=30)  # クロージング
    # cv2.imshow("mask_new",hsv_mask)

    # 輪郭取得
    contours, _ = cv2.findContours(
        hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    [h,w]=hsv_mask.shape
    min_area=h//10*w//10
    cut_points=[] 
    for i, cnt in enumerate(contours):
        arclen = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, arclen*0.02, True)
        area = abs(cv2.contourArea(approx))
        if approx.shape[0] == 4 and area > min_area  and cv2.isContourConvex(approx):
            rcnt = approx.reshape(-1, 2)
            
            return rcnt
    
    return []
def mask_sort(mask):
    new_mask=[[0,0],[0,0],[0,0],[0,0]]

    sum=np.array(mask).sum(1)
    mask=list(mask)
    if sum.argmin()>sum.argmax():
        new_mask[0]=mask.pop(sum.argmin())
        new_mask[3]=mask.pop(sum.argmax())
    else:
        new_mask[3]=mask.pop(sum.argmax())
        new_mask[0]=mask.pop(sum.argmin())
        

    if mask[0][0]>mask[1][0]:
        new_mask[1]=mask[0]
        new_mask[2]=mask[1]
    else:
        new_mask[1]=mask[1]
        new_mask[2]=mask[0]
    return new_mask

def get_dst(field_points,mask,dst,max,a=40):
    def_points=[[field_points[0][0]+a,field_points[0][1]+a]
                ,[field_points[1][0]-a,field_points[1][1]-a]] 
    mask=mask_sort(mask)
    sum=abs(mask[0][0]-def_points[0][0])
    sum+=abs(mask[2][0]-def_points[0][0])
    sum+=abs(mask[3][0]-def_points[1][0])
    sum+=abs(mask[1][0]-def_points[1][0])
    # # 縦
    sum+=abs(mask[0][1]-def_points[0][1])//2
    sum+=abs(mask[2][1]-def_points[0][1])//2
    sum+=abs(mask[3][1]-def_points[1][1])//2
    sum+=abs(mask[1][1]-def_points[1][1])//2
    if sum<25:
        return dst,True
    if mask[0][0]>def_points[0][0]:
        dst[0][0]-=8
    elif mask[0][0]<def_points[0][0]:
        dst[0][0]+=1
    
    

    if mask[2][0]>def_points[0][0]:
        dst[2][0]-=8
    elif mask[2][0]<def_points[0][0]:
        dst[2][0]+=1

    if mask[3][0]<def_points[1][0]:
        dst[3][0]+=8
    elif mask[3][0]>def_points[1][0]:
        dst[3][0]-=1

    if mask[1][0]<def_points[1][0]:
        dst[1][0]+=8
    elif mask[1][0]>def_points[1][0]:
        dst[1][0]-=1
    
    #縦
    if mask[0][1]>def_points[0][1]:
        dst[0][1]-=2
    elif mask[0][1]<def_points[0][1]:
        dst[0][1]+=1
        
    
    if mask[2][1]<def_points[1][1]:
        dst[2][1]+=2
        if dst[3][1]>max:
            dst[3][1]=max
    elif mask[2][1]>def_points[1][1]:
        dst[2][1]-=1

    if mask[3][1]<def_points[1][1]:
        dst[3][1]+=2
        if dst[3][1]>max:
            dst[3][1]=max
    elif mask[3][1]>def_points[1][1]:
        dst[3][1]-=1

    if mask[1][1]>def_points[0][1]:
        dst[1][1]-=2
    elif mask[1][1]<def_points[0][1]:
        dst[1][1]+=1
    dst[dst<0]=0
    return dst,False
#緑領域抽出
def get_green(path):
    if type(path) is str:
        img = cv2.imread(path)
    else:
        img=path.copy()
    height, width, channels = img.shape[:3]
    hsvLower = np.array([50, 100, 100])    # 抽出する色の下限(HSV)
    hsvUpper = np.array([110, 255, 255])    # 抽出する色の上限(HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 画像をHSVに変換
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSVからマスクを作成
    kernel = np.ones((3, 3), np.uint8)
    hsv_mask = cv2.morphologyEx(
        hsv_mask, cv2.MORPH_OPEN, kernel, iterations=3)  # クロージング
    kernel = np.ones((3, 3), np.uint8)
    hsv_mask = cv2.morphologyEx(
        hsv_mask, cv2.MORPH_CLOSE, kernel, iterations=10)  # クロージング
    
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
                return rcnt    
    return []
    