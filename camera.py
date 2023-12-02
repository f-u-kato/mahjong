import cv2
from screeninfo import get_monitors

import numpy as np
import random
import time
import datetime
import pandas as pd
import os
import glob
import re

import src.out_func.draw_img as draw
import src.eval.mahjong_eval as eval
import src.get_func.get_area as area
import src.get_func.get_img as get
import src.eval.calculation as mahjong_calculation
import src.out_func.play_music as music
import output_video as ov
import src.out_func.transform_video as trans

import concurrent.futures 



#結果出力用
def print_hand_result(hand_result, agari):
    result = [
        f"{hand_result.han} han, {hand_result.fu} fu",
        f"{hand_result.cost['main']}, {hand_result.cost['additional']}",
        f"{hand_result.yaku}",
        f"agarihai: {agari}"
    ]

    return result

PLAY_BGM = glob.glob("./music/BGM/*")
RIICHI_BGM = glob.glob("./music/riichi/*")
TRIGGER_SE = [r'.\music\効果音1.mp3']
LOAD_SE = r'.\music\loading.mp3'
AGARI_SE = [r'.\music\和太鼓でドン.mp3', r'.\music\和太鼓でドドン.mp3']
RYOUKYOKU_SE = r'.\music\しょげる.mp3'
POINT_SE = [r'.\music\平手打ち1.mp3', r'.\music\剣で斬る2.mp3', r'.\music\剣で斬る1.mp3', r'.\music\剣で斬る3.mp3', r'.\music\剣で斬る4.mp3', r'.\music\剣で斬る6.mp3']
AGARI_IMAGES = ['./material/points/mangan.png', './material/points/haneman.png',
                './material/points/baiman.png', './material/points/3bai.png', './material/points/yakuman.png']
BACK_MOVIES = glob.glob("./material/back/*")
AGARI_VIDEOS = ['./material/満貫.mp4', './material/跳満.mp4', './material/倍満.mp4', './material/三倍満.mov']
YAKUMAN_VIDEOS = ['./material/役満1.mp4', './material/役満2.mov', './material/役満3.mp4']
TRIGGER_VIDEOS = ['./material/trigger/0.mp4', './material/trigger/1.mp4']
RIICHI_SE = ["./music/riichi.mp3"]
RIICHI_VIDEO = "./material/riichi.mp4"



MAHJONG_CLASSES = ("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                   "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                   "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
                   "ton", "nan", "sha", "pe",
                   "haku", "hatsu", "chun",
                   "aka_5m", "aka_5p", "aka_5s",
                   "ura")
MAHJONG_CLASSES_NUMBER = {"1m":"1", "2m":"2", "3m":"3", "4m":"4", "5m":"5", "6m":"6", "7m":"7", "8m":"8", "9m":"9",
                          "1p":"1", "2p":"2", "3p":"3", "4p":"4", "5p":"5", "6p":"6", "7p":"7", "8p":"8", "9p":"9",
                          "1s":"1", "2s":"2", "3s":"3", "4s":"4", "5s":"5", "6s":"6", "7s":"7", "8s":"8", "9s":"9",
                          "ton":"1", "nan":"2", "sha":"3", "pe":"4",
                          "haku":"5", "hatsu":"6", "chun":"7",
                          "aka_5m":"0", "aka_5p":"0", "aka_5s":"0",
                          "ura":"-1"}


def read_trigger(cap, field_points, size, cM, ton_player, m, round_wind, honba, dst, player_points, reduction=1, save_movie=None, effect=None):
    # 動画の再生速度
    speed = 1
    # トリガー動画の読み込み
    trigger = cv2.VideoCapture(TRIGGER_VIDEOS[random.randint(0, len(TRIGGER_VIDEOS)-1)])
    # 情報と領域の投影
    img = draw.draw_rect_movie(field_points, trigger, size, img=None, reduction=reduction)
    img = draw.draw_kaze(field_points, ton_player, img=img, reduction=reduction)
    img = draw.draw_honba(field_points, ton_player, round_wind, honba, img=img, reduction=reduction)
    img = draw.draw_riichi(field_points, img=img, reduction=reduction)
    img = draw.draw_player_points(field_points, player_points, img=img, reduction=reduction)
    sM = ov.show_img(img, m, field_points, dst=dst, reduction=reduction)
    cv2.waitKey(1)
    count = 0
    # 背景動画の読み込み
    rand = random.randint(0, len(PLAY_BGM)-1)
    video = cv2.VideoCapture(BACK_MOVIES[rand])
    save_dir = "./save_riichi/0"
    os.makedirs(save_dir,exist_ok=True)
    while (cap.isOpened()):
        # カメラ映像の取得
        ret, im = cap.read()
        im = trans.transform_camera(im, M=cM)


        # カメラ映像の表示
        im = draw.draw_rect2(field_points, size, im)
        im = draw.draw_riichi2(field_points, im, size)
        cv2.imshow("Camera", cv2.resize(im, (1920, 1080)))

        # 立直保存
        if count % 10 == 0:
            for i in range(4):
                riichi_image=get.get_riichi(field_points, i, im)
                riichi_image=eval.padding_riichi(riichi_image)
                cv2.imwrite(f"{save_dir}/riichi{i}_{count}.png", riichi_image)
            print("save")
        count += 1
            

        # 背景動画の投影
        img = draw.loop_movie(field_points, video, size, ton_player, reduction=reduction, speed=speed)
        # トリガー動画の投影
        img = draw.draw_rect_movie(field_points, trigger, size, img=img, reduction=reduction)
        # 情報の投影
        img = draw.draw_kaze(field_points, ton_player, img=img, reduction=reduction)
        img = draw.draw_honba(field_points, ton_player, round_wind, honba, img=img, reduction=reduction)
        img = draw.draw_riichi(field_points, img=img, reduction=reduction)
        img = draw.draw_player_points(field_points, player_points, img=img, reduction=reduction)
        ov.show_img(img, m, field_points, M=sM, reduction=reduction)
        c = cv2.waitKey(1)

        # 流局
        if c == ord('q'):
            return
        elif c == ord('p'):
            save_dir = "./save_riichi/1"
            print(save_dir)
            os.makedirs(save_dir,exist_ok=True)
            cv2.waitKey()
            

    return 


def camera():
    now = datetime.datetime.now()
    time_df = pd.DataFrame()

    # カメラの設定
    m = get_monitors()[1]
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)

    # 動画保存用設定
    effect = None
    # effect = save_video(cap, "./save_movie/all_effect.mp4")
    save_movie = None
    # save_movie = save_video(cap, "./save_movie/all_movie.mp4")

    # モデルを読み込んでおく
    _ = eval.trigger_eval(np.zeros([100, 100, 3], dtype=np.uint8))
    _, _, _ = eval.win_eval(np.zeros([100, 100, 3], dtype=np.uint8), 0.9)

    # カメラ調節
    while (cap.isOpened()):
        ret, im = cap.read()
        if ret:
            def_points = area.get_green(im)
            if len(def_points) > 0:
                # 判定領域の表示
                new_im = im.copy()
                cv2.polylines(new_im, [def_points], True, (0, 0, 255), 4)
                cv2.imshow("Camera", cv2.resize(new_im, (1920, 1080)))

                if save_movie is not None:
                    save_movie.write(cv2.resize(im, (1920, 1080)))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    im_middle = [im.shape[0]//2, im.shape[1]//2]
    field_points = [[im_middle[1]-im_middle[0]+10, 10], [im_middle[1]+im_middle[0]-10, im.shape[0]-10]]

    # カメラ
    def_points = area.mask_sort(def_points)
    print("field", field_points)
    print("def", def_points)
    save_im, cM = trans.transform_camera(im, field_points, src=def_points)
    save_im[:field_points[0][1], :] = 0  # 上部
    save_im[field_points[1][1]:, :] = 0  # 下部
    save_im[:, :field_points[0][0]] = 0  # 左側
    save_im[:, field_points[1][0]:] = 0  # 右側

    # 投影設定
    size = im.shape
    middle = [size[1]//2, size[0]//2]

    dst = np.float32([np.array([middle[0]-size[1]//10, middle[1]//30]), np.array([middle[0]+size[1]//3, middle[1]//30]),
                      np.array([middle[0]-size[1]//10, middle[1]*19//10]), np.array([middle[0]+size[1]//3, middle[1]*19//10])])

    reduction = size[0]/size[0]
    min_field = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                 [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    cv2.destroyAllWindows()
    # 領域の投影
    img = np.zeros(size, np.uint8)
    cv2.rectangle(img, min_field[0], min_field[1], (255, 0, 0), -1)
    _ = ov.show_img(img, m, min_field, dst=dst)
    cv2.waitKey(500)
    a = 50
    def_points = [[field_points[0][0]+a, field_points[0][1]+a], [field_points[1][0]-a, field_points[1][1]-a]]

    # 投影調節
    sum = 30
    isBreak = False
    while (cap.isOpened()):
        ret, im = cap.read()
        im = trans.transform_camera(im, M=cM)
        if save_movie is not None:
            save_movie.write(cv2.resize(im, (1920, 1080)))
        im[:field_points[0][1], :] = 0  # 上部
        im[field_points[1][1]:, :] = 0  # 下部
        im[:, :field_points[0][0]] = 0  # 左側
        im[:, field_points[1][0]:] = 0  # 右側
        mask = area.get_rect(im, save_im)
        cv2.rectangle(im, def_points[0], def_points[1], (0, 255, 0), 3)
        if len(mask) > 0:
            cv2.polylines(im, [mask], True, (0, 0, 255), 3)
            dst, isBreak = area.get_dst(field_points, mask, dst, max=size[0])
        cv2.imshow("mask", cv2.resize(im, (1920, 1080)))
        img = np.zeros(size, np.uint8)
        cv2.rectangle(img, min_field[0], min_field[1], (255, 0, 0), -1)
        if effect is not None:
            effect.write(cv2.resize(img, (1920, 1080)))
        _ = ov.show_img(img, m, min_field, dst=dst)

        if cv2.waitKey(500) & 0xFF == ord('q') or isBreak:
            break

    cv2.destroyAllWindows()

    # 初期設定
    size = im.shape
    round_wind = 0
    isContinue = True
    player_points = [25000, 25000, 25000, 25000]
    ton_player = 0
    honba = 0
    kyotaku = 0

    # ゲーム開始
    min_size=(540, 960, 3)
    reduction = size[0]/min_size[0]
    read_trigger(cap, field_points, size, cM, ton_player, m, round_wind, honba, dst=dst,
                                player_points=player_points, reduction=reduction, save_movie=save_movie, effect=effect)
    return
    

def hand_camera():
    m=get_monitors()[1]
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
    cap.set(cv2.CAP_PROP_FOCUS,0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,3840)
     # カメラ調節
    while(cap.isOpened()):
        ret, im = cap.read()
        field_points=area.get_green(im)
        color = (0, 255, 0)
        # im,_=transform_camera(im,dst=field_points)

        if len(field_points) > 0:
            new_im=im.copy()
            cv2.rectangle(new_im, field_points[0], field_points[1], color,3)
            cv2.imshow("Camera", cv2.resize(new_im,(1920,1080)))
            size=im.shape
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #-卓領域検出
    while(cap.isOpened()):
        ret, im = cap.read()
        field_points=area.get_green(im)
        if len(field_points) > 0:
            break
    _,cM=transform_camera(im,dst=field_points)
    img=draw.draw_player_rect(field_points,2,size)
    sM=show_img(img,m,field_points)
    count=33
    while(1):
        ret, im = cap.read()
        im=transform_camera(im,M=cM)
        cv2.imshow('Camera',cv2.resize(im,(1920,1080)))
        c=cv2.waitKey(1)
        if c == ord('q'):
            break
        elif c==ord("p"):
            img=get.get_hand(field_points,2,im,size)
            cv2.imwrite(f"./time/hand{count}.png",img)
            print(count)
            count+=1
        

def get_max_int(folder_path):
    max_int = -1 

    # フォルダ内のファイルを走査
    for filename in os.listdir(folder_path):
        # ファイル名から数字のみを抽出
        match = re.search(r'\d+', filename)
        if match is not None:
            # 整数に変換して最大値を更新
            file_int = int(match.group())
            if file_int > max_int:
                max_int = file_int
    return max_int

def riichi_camera():
    cap,m,dst,ton_player,field_points,cM,size,round_wind,honba=setting_camera()
    min_size=(540,960,3)
    reduction=size[0]/min_size[0]
    img=draw.draw_riichi(field_points,img=None,reduction=reduction)
    print(type(img))
    sM=ov.show_img(img,m,field_points,dst=dst,reduction=reduction)
    print('riichi',img.shape)
    sM=ov.show_img(img,m,field_points,dst=dst,reduction=reduction)
    for i in range(2):
        os.makedirs("./riichi/"+str(i),exist_ok=True)
        count=get_max_int("./riichi/"+str(i))
        img=draw.draw_riichi(field_points,reduction=reduction)
        ov.show_img(img,m,field_points,M=sM,reduction=reduction)
        cv2.waitKey()
        print("start")
        trigger=cv2.VideoCapture(TRIGGER_VIDEOS)
        rand=0
        video=cv2.VideoCapture(BACK_MOVIES[rand])
        change=1
        while(1):
            ret, im = cap.read()
            im=ov.transform_camera(im,M=cM)
            cv2.imshow('Camera',cv2.resize(im,(1920,1080)))
            
            
            img=draw.loop_movie(field_points,video,size,ton_player,reduction=reduction,speed=1)
            img = draw.draw_rect_movie(field_points,trigger,size,img=img,reduction=reduction)
            img = draw.draw_kaze(field_points,ton_player,img=img,reduction=reduction)
            img = draw.draw_honba(field_points,ton_player,round_wind,honba,img=img,reduction=reduction)
            img=draw.draw_riichi(field_points,img=img,reduction=reduction)
            ov.show_img(img,m,field_points,M=sM,reduction=reduction)
            c=cv2.waitKey(1)
            if count%10==0:
                print(count)
                get_img=get.get_riichi(field_points,im,size)
                cv2.imwrite(f"./riichi/{i}/{count}.png",get_img)
            if change%5000==0:
                rand+=1
                if rand>=len(BACK_MOVIES):
                    rand=0
                video=cv2.VideoCapture(BACK_MOVIES[rand])
            if c == ord('q'):
                print("end_0")
                break
            count+=1
            change+=1
        video.release()
        trigger.release()
def trigger_camera():
    cap,m,dst,ton_player,field_points,cM,size,round_wind,honba=setting_camera()
    img=draw.draw_rect(field_points,img=None)
    _=ov.show_img(img,m,field_points,dst=dst)
    for i in range(2):
        os.makedirs("./trigger2/"+str(i),exist_ok=True)
        count=get_max_int("./trigger2/"+str(i))
        ret, im = cap.read()
        im=ov.transform_camera(im,M=cM)
        im=draw.draw_rect2(field_points,img=im)
        cv2.imshow("all",cv2.resize(im,(1920,1080)))
        cv2.waitKey()
        print("start")
        while(1):
            ret, im = cap.read()
            im=ov.transform_camera(im,M=cM)
            c=cv2.waitKey(1)
            if count%10==0:
                print(count)
                get_img=get.get_trigger(field_points,2,im)
                cv2.imshow("trigger",get_img)
                cv2.imwrite(f"./trigger2/{i}/{count}.png",get_img)
            if c == ord('q'):
                print("end_0")
                break
            count+=1
def setting_camera():
    m=get_monitors()[1]
    ton_player=1
    honba=0
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
    cap.set(cv2.CAP_PROP_FOCUS,0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,3840)
    
    
    
    # カメラ調節
    while(cap.isOpened()):
        ret, im = cap.read()
        if ret:
            def_points=area.get_green(im)
            color = (0, 0, 255)
            # im,_=transform_camera(im,dst=field_points)
            if len(def_points) > 0:
                new_im=im.copy()
                
                cv2.polylines(new_im,[def_points],True,color,4)
                cv2.imshow("Camera", cv2.resize(new_im,(1920,1080)))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    im_middle=[im.shape[0]//2,im.shape[1]//2]
    field_points=[[im_middle[1]-im_middle[0]+10,10],[im_middle[1]+im_middle[0]-10,im.shape[0]-10]]
    #カメラ
    def_points=[def_points[0],def_points[3],def_points[1],def_points[2]]
    save_im,cM=ov.transform_camera(im,field_points,src=def_points)
    save_im[:field_points[0][1], :] = 0  # 上部
    save_im[field_points[1][1]:, :] = 0  # 下部
    save_im[:, :field_points[0][0]] = 0  # 左側
    save_im[:, field_points[1][0]:] = 0  # 右側
    
    #投影設定
    size=im.shape
    middle=[size[1]//2,size[0]//2]
    # dst=np.float32([np.array([middle[0]-150,middle[1]-250]),np.array([middle[0]+400,middle[1]-250]),
    #                 np.array([middle[0]-150,middle[1]+250]),np.array([middle[0]+400,middle[1]+250])])
    dst=np.float32([np.array([middle[0]-size[1]//6,0]),np.array([middle[0]+size[1]//3,0]),
                    np.array([middle[0]-size[1]//6,middle[1]*2]),np.array([middle[0]+size[1]//3,middle[1]*2])])
    
    reduction=size[0]/size[0]
    min_field=([int(field_points[0][0]//reduction),int(field_points[0][1]//reduction)],[int(field_points[1][0]//reduction),int(field_points[1][1]//reduction)])
    cv2.destroyAllWindows()
    #領域の投影
    img=np.zeros(size,np.uint8)
    cv2.rectangle(img, min_field[0], min_field[1], (255, 0, 0),-1)
    _=ov.show_img(img,m,min_field,dst=dst)
    cv2.waitKey(500)
    a=50
    def_points=[[field_points[0][0]+a,field_points[0][1]+a]
                ,[field_points[1][0]-a,field_points[1][1]-a]] 
    
    #投影調節
    isBreak=False
    while(cap.isOpened()):
        ret, im = cap.read()
        im=ov.transform_camera(im,M=cM)
        im[:field_points[0][1], :] = 0  # 上部
        im[field_points[1][1]:, :] = 0  # 下部
        im[:, :field_points[0][0]] = 0  # 左側
        im[:, field_points[1][0]:] = 0  # 右側
        mask=area.get_rect(im,save_im)
        cv2.rectangle(im,def_points[0],def_points[1],(0,255,0),3)
        if len(mask)>0:
            cv2.polylines(im,[mask],True,color,3)
            dst,isBreak=area.get_dst(field_points,mask,dst)
        cv2.imshow("mask", cv2.resize(im,(1920,1080)))
        img=np.zeros(size,np.uint8)
        cv2.rectangle(img, min_field[0], min_field[1], (255, 0, 0),-1)
        _=ov.show_img(img,m,min_field,dst=dst)
        
        if cv2.waitKey(500) & 0xFF == ord('q') or isBreak:
            break

    cv2.destroyAllWindows()
    size=im.shape
    round_wind=0
    

    return cap,m,dst,ton_player,field_points,cM,size,round_wind,honba


def field_camera():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
    cap.set(cv2.CAP_PROP_FOCUS,0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,3840)
    # カメラ調節
    while(cap.isOpened()):
        ret, im = cap.read()
        if ret:
            def_points=area.get_green(im)
            color = (0, 0, 255)
            # im,_=transform_camera(im,dst=field_points)
            if len(def_points) > 0:
                new_im=im.copy()
                
                cv2.polylines(new_im,[def_points],True,color,4)
                cv2.imshow("Camera", cv2.resize(new_im,(960,540)))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    im_middle=[im.shape[0]//2,im.shape[1]//2]
    field_points=[[im_middle[1]-im_middle[0]+10,10],[im_middle[1]+im_middle[0]-10,im.shape[0]-10]]
    #カメラ
    def_points=[def_points[0],def_points[3],def_points[1],def_points[2]]
    _,cM=ov.transform_camera(im,field_points,src=def_points)
    os.makedirs("./save_movie",exist_ok=True)
    save=save_video(cap,"./save_movie/field.mp4")
    while(cap.isOpened()):
        ret, im = cap.read()
        im=ov.transform_camera(im,M=cM)
        im=cv2.resize(im,(1920,1080))
        save.write(im)
        im=cv2.resize(im,(960,540))
        cv2.imshow("all",im)
        
        c=cv2.waitKey(1)

        if c == ord('q'):
            print("end_0")
            break
    cap.release()
    save.release()
def save_video(camera,name):
    fps = int(camera.get(cv2.CAP_PROP_FPS))                    # カメラのFPSを取得
    w = int(1920)              # カメラの横幅を取得
    h = int(1080)             # カメラの縦幅を取得
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        # 動画保存時のfourcc設定（mp4用）
    video = cv2.VideoWriter(name, fourcc, fps, (w, h))  # 動画の仕様（ファイル名、fourcc, FPS, サイズ）
    return video
if __name__ == '__main__':
    camera()