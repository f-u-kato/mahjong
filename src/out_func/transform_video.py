#カメラの補正
import cv2
import numpy as np


def transform_camera(path,field_points=None,src=None,M=None):
    img=path.copy()
    height,width,_=img.shape
    if M is None:
        dst=[field_points[0],[field_points[1][0],field_points[0][1]]
             ,[field_points[0][0],field_points[1][1]],field_points[1]]
        dst=np.float32(dst)
        src=np.float32(src)
        
        # 変換行列
        M = cv2.getPerspectiveTransform(src, dst)
    
        # 射影変換・透視変換する
        output = cv2.warpPerspective(img, M,(width, height))
        return output,M
    else:
        output = cv2.warpPerspective(img, M,(width, height))
        return output

#投影の補正
def transform_img(path,dst=None,field_points=None,M=None):
    img=path.copy()
    
    height,width,_=img.shape
    if M is None:
        src=[field_points[0],[field_points[1][0],field_points[0][1]]
             ,[field_points[0][0],field_points[1][1]],field_points[1]]
        src=np.float32(src)
        # 変換行列
        M = cv2.getPerspectiveTransform(src, dst)
    
        # 射影変換・透視変換する
        output = cv2.warpPerspective(img, M,(width, height))
        return output,M
    else:
        output = cv2.warpPerspective(img, M,(int(width), int(height)))
        return output