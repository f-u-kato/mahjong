import cv2
from screeninfo import get_monitors

import numpy as np
import random
import time
import datetime
import pandas as pd
import os
import glob

import src.out_func.draw_img as draw
import src.eval.mahjong_eval as eval
import src.get_func.get_area as area
import src.get_func.get_img as get
import src.eval.calculation as mahjong_calculation
import src.out_func.play_music as music
import src.out_func.transform_video as trans
import src.out_func.camera as camera

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
LOAD_SE=r'.\music\loading.mp3'
AGARI_SE=[r'.\music\和太鼓でドン.mp3',r'.\music\和太鼓でドドン.mp3']
RYOUKYOKU_SE=r'.\music\しょげる.mp3'
POINT_SE=[r'.\music\平手打ち1.mp3',r'.\music\剣で斬る2.mp3',r'.\music\剣で斬る1.mp3',r'.\music\剣で斬る3.mp3',r'.\music\剣で斬る4.mp3',r'.\music\剣で斬る6.mp3']
AGARI_IMAGES = ['./material/points/mangan.png','./material/points/haneman.png','./material/points/baiman.png','./material/points/3bai.png','./material/points/yakuman.png']
BACK_MOVIES=glob.glob("./material/back/*")
AGARI_VIDEOS=['./material/満貫.mp4','./material/跳満.mp4','./material/倍満.mp4','./material/三倍満.mov']
YAKUMAN_VIDEOS=['./material/役満1.mp4','./material/役満2.mov','./material/役満3.mp4']
TRIGGER_VIDEOS=['./material/trigger/0.mp4','./material/trigger/1.mp4']
RIICHI_SE=["./music/riichi.mp3"]
RIICHI_VIDEO="./material/riichi.mp4"



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

def get_wind(player_num,ton_num):
    num=player_num-ton_num
    if num<0:
        num+=4
    return num


def check_tile(field_points,im,size=(2160,3840,3),threshold=0.8):
    hai_size=max(field_points[1][0]-field_points[0][0],field_points[1][1]-field_points[0][1])//15
    hai_point=[hai_size*2+50,hai_size*2]
    field_size=[field_points[1][0]-field_points[0][0],field_points[1][1]-field_points[0][1]]
    for i in range(4):
        img=get.get_trigger(field_points,i+1,im)
        if in_hai(img):
            classes = eval.trigger_eval(img)
            if classes==0:
                return i+1
    return 0

#牌があるかの判定
def in_hai(img,a=0.9):
    height, width, channels = img.shape[:3]
    hsvLower = np.array([30, 50, 40])    # 抽出する色の下限(HSV)
    hsvUpper = np.array([90, 255, 255])    # 抽出する色の上限(HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 画像をHSVに変換
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSVからマスクを作成
    if height*width*a<np.sum(hsv_mask//255) or height*width*0.1>np.sum(hsv_mask//255):
        return False
    else:
        return True


def show_img(img,size_data=None,field_points=None,dst=None,M=None,reduction=1):
    
    # ウィンドウを作成する
    cv2.namedWindow("Projector Output", cv2.WINDOW_NORMAL)

    # 画像を表示する
    if M is None:
        new_dst=dst.copy()
        new_dst/=reduction
        draw_points=([int(field_points[0][0]//reduction),int(field_points[0][1]//reduction)],[int(field_points[1][0]//reduction),int(field_points[1][1]//reduction)])
        new_im,M=trans.transform_img(img,dst=new_dst,field_points=draw_points)
        cv2.imshow("Projector Output", new_im)
        cv2.moveWindow('Projector Output',size_data.x,size_data.y)
        # プロジェクターに接続する
        cv2.setWindowProperty("Projector Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow("Projector Output", size_data.width, size_data.height)
        return M
    else:
        cv2.imshow("Projector Output", trans.transform_img(img,M=M))
        cv2.moveWindow('Projector Output',size_data.x,size_data.y)
        # プロジェクターに接続する
        cv2.setWindowProperty("Projector Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow("Projector Output", size_data.width, size_data.height)

def save_video(camera,name):
    fps = int(camera.get(cv2.CAP_PROP_FPS))                    # カメラのFPSを取得
    w = int(1920)              # カメラの横幅を取得
    h = int(1080)             # カメラの縦幅を取得
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        # 動画保存時のfourcc設定（mp4用）
    video = cv2.VideoWriter(name, fourcc, fps, (w, h))  # 動画の仕様（ファイル名、fourcc, FPS, サイズ）
    return video


def read_trigger(cap, field_points, size, cM, ton_player, m, round_wind, honba,dst,reduction=1,save_movie=None,effect=None):
    isRiichi=False
    r_flag=True
    rcount=0
    r_video=None
    speed=1
    trigger=cv2.VideoCapture(TRIGGER_VIDEOS[random.randint(0,len(TRIGGER_VIDEOS)-1)])
    img = draw.draw_rect_movie(field_points,trigger,size,img=None,reduction=reduction)
    img = draw.draw_kaze(field_points,ton_player,img=img,reduction=reduction)
    img = draw.draw_honba(field_points,ton_player,round_wind,honba,img=img,reduction=reduction)
    img=draw.draw_riichi(field_points,img=img,reduction=reduction)
    sM=show_img(img,m,field_points,dst=dst,reduction=reduction)
    cv2.waitKey(1)
    #音楽再生
    rand=random.randint(0,len(PLAY_BGM)-1)
    music.loop_music(PLAY_BGM[rand])
    count = 0
    video=cv2.VideoCapture(BACK_MOVIES[rand])
    # i=0
    # os.makedirs("./riichi/"+str(i),exist_ok=True)
    # count=camera.get_max_int("./riichi/"+str(i))
    while(cap.isOpened()):
        ret, im = cap.read()
        im = trans.transform_camera(im,M=cM)
        if save_movie is not None:
            save_movie.write(cv2.resize(im,(1920,1080)))
        
        win_player = check_tile(field_points, im, size)
        if win_player > 0:
            print('check')
            break
        im = draw.draw_rect2(field_points,size,im)
        im=draw.draw_riichi2(field_points,im,size)
        cv2.imshow("Camera", cv2.resize(im,(1920,1080)))
        
        if not isRiichi:
            if eval.riichi_eval(get.get_riichi(field_points,im))==1:
                rcount+=1
                if rcount>5:
                    music.stop_music()
                    r_video=cv2.VideoCapture(RIICHI_VIDEO)
                    music.play_music(RIICHI_SE[0])
                    speed*=2
                    isRiichi=True
                else:
                    isRiichi=False
            else:
                rcount=0

        img=draw.loop_movie(field_points,video,size,ton_player,reduction=reduction,speed=speed)
        
        if isRiichi and r_flag:
            _,_=r_video.read()
            _,_=r_video.read()
            tmp_img = draw.back_place(r_video,img,field_points,ton_player,reduction=reduction,skelton=True)
            if tmp_img is None:
                r_flag=False
                r_video.release()
                music.loop_music(RIICHI_BGM[rand])
            else:
                img=tmp_img
        img = draw.draw_rect_movie(field_points,trigger,size,img=img,reduction=reduction)
        img = draw.draw_kaze(field_points,ton_player,img=img,reduction=reduction)
        img = draw.draw_honba(field_points,ton_player,round_wind,honba,img=img,reduction=reduction)

        img=draw.draw_riichi(field_points,img=img,reduction=reduction)
        if effect is not None:
            effect.write(cv2.resize(img,(1920,1080)))
        show_img(img,m,field_points,M=sM,reduction=reduction)
        c = cv2.waitKey(1)
        # if count%10==0:
        #     print(count)
        #     get_img=get.get_riichi(field_points,im,size)
        #     cv2.imwrite(f"./riichi/{i}/{count}.png",get_img)
        if c == ord('q'):
            music.stop_music()
            music.play_music(RYOUKYOKU_SE)
            return -1
        elif c == ord('p'):
            music.stop_music()
            music.play_music(RYOUKYOKU_SE)
            return -2
        # elif c == ord('c'):
        #     print('stop')
        #     i=1
        #     os.makedirs("./riichi/"+str(i),exist_ok=True)
        #     count=camera.get_max_int("./riichi/"+str(i))
        #     cv2.waitKey()
        #     print('start')
        # count+=1
    return win_player

def wait_no_wintile(field_points,win_player,size,dst,cap,cM,m,im=None,reduction=1,save_movie=None,effect=None):
    img=draw.draw_player_rect(field_points,win_player,size,first=True,img=im,reduction=reduction)
    show_img(img,m,field_points,dst=dst,reduction=reduction)
    while(cap.isOpened()):
        ret, im = cap.read()
        im=trans.transform_camera(im,M=cM)
        if save_movie is not None:
            save_movie.write(cv2.resize(im,(1920,1080)))
        if effect is not None:
            effect.write(cv2.resize(img,(1920,1080)))
        if not in_hai(get.get_wintile(field_points,win_player,im,size)):
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            music.stop_music()
            break

#上がり牌のチェック
def read_wintile(field_points,win_player,size,cap,cM,ton_player,m,dst,is_eval_draw=True,reduction=1,save_movie=None,effect=None):
    music.stop_music()
    music.play_music(TRIGGER_SE[random.randint(0,len(TRIGGER_SE)-1)])
    
    def_img=draw.draw_player_rect(field_points,win_player,size,reduction=reduction)
    def_img=draw.draw_kaze(field_points,ton_player,img=def_img,reduction=reduction)

    sM=show_img(def_img,m,field_points,dst=dst,reduction=reduction)
    img=def_img.copy()
    isFirst=True
    
    lose_player=0
    
    while(cap.isOpened()):
        show_img(def_img,m,field_points,M=sM,reduction=reduction)
        for i in range(500):
            if isFirst:
                isFirst=False
                break
            c2=cv2.waitKey(1)
            if c2 == ord('q'):
                return -1,-1
        ret, im = cap.read()
        show_img(img,m,field_points,M=sM,reduction=reduction)
        cv2.waitKey(1)
        im=trans.transform_camera(im,M=cM)
        new_im=im.copy()
        new_im=draw.draw_player_rect2(field_points,win_player,size,new_im)
        cv2.imshow("Camera", cv2.resize(new_im,(1920,1080)))
        hai_images=[]
        for i in range(4):
            hai_images.append(get.get_wintile(field_points,i+1,im,size))
        
        win_evals=eval.multi_win_eval(hai_images,0.8)
        for i,win_eval in enumerate(win_evals):
            [win_class,win_score,win_box]=win_eval
            if len(win_class)>0 and win_class!=37:
                lose_player=i+1
                print('set ok')
                music.play_music(LOAD_SE)
                break
        if lose_player>0:
            break
        #結果の表示
        if is_eval_draw:
            hai_img=get.get_hand(field_points,win_player,im,size)
            hand_classes,hand_scores,hand_boxes=eval.hand_eval(hai_img,0.3)
            hai_img=get.get_dora(field_points,win_player,im,size)
            dora_classes,dora_scores,dora_boxes=eval.dora_eval(hai_img,0.5)
            hai_img=get.get_naki(field_points,win_player,im,size)
            naki_classes,naki_scores,naki_boxes=eval.naki_eval(hai_img,0.5)
            img=draw.draw_hand(field_points,hand_classes,hand_boxes,win_player,size,def_img.copy(),reduction=reduction)
            img=draw.draw_dora(field_points,dora_classes,dora_boxes,win_player,size,img,reduction=reduction)
            img=draw.draw_naki(field_points,naki_classes,naki_boxes,win_player,size,img,reduction=reduction)
        c=cv2.waitKey(1)
        if c == ord('q'):
            return -1,-1,0
        if c == ord('p'):
            return -2,-2,0
    return win_class,win_box,lose_player




def draw_movie(field_points,size,m,cap,win_player,cM,agari,dst,min_size=(540,960,3),save_movie=None,effect=None):
    
    reduction=size[0]/min_size[0]
    min_field=field_points
    img=np.zeros(min_size,np.uint8)
    min_sM=show_img(img,m,field_points,dst=dst,reduction=reduction)
    if agari<4:
        video=AGARI_VIDEOS[agari]
        video=cv2.VideoCapture(AGARI_VIDEOS[agari])
        count=0
        while(cap.isOpened()):
            ret, im = cap.read()
            im=trans.transform_camera(im,M=cM)
            if save_movie is not None:
                save_movie.write(cv2.resize(im,(1920,1080)))
            
            cv2.imshow("Camera", cv2.resize(im,(1920,1080)))
            img=np.zeros(min_size,np.uint8)
            frame=draw.back_place(video,img,min_field,win_player,count,reduction=reduction)
            if frame is None:
                video.release()
                return
            if effect is not None:
                effect.write(cv2.resize(frame,(1920,1080)))
            show_img(frame,m,min_field,M=min_sM,reduction=reduction)
            c=cv2.waitKey(1)
            count+=2
            if c == ord('q') :
                video.release()
                return
    else:
        for i in range(len(YAKUMAN_VIDEOS)):
            video=YAKUMAN_VIDEOS[i]
            video=cv2.VideoCapture(video)
            if i==2:
                music.play_music("./music/和風ジングル.mp3")
            count=0
            while(cap.isOpened()):
                ret, im = cap.read()
                im=trans.transform_camera(im,M=cM)
                if save_movie is not None:
                    save_movie.write(cv2.resize(im,(1920,1080)))
                cv2.imshow("Camera", cv2.resize(im,(1920,1080)))
                img=np.zeros(min_size,np.uint8)
                frame=draw.back_place(video,img,min_field,win_player,count,reduction=reduction)
                if frame is None:
                    break
                if effect is not None:
                    effect.write(cv2.resize(frame,(1920,1080)))
                show_img(frame,m,min_field,M=min_sM,reduction=reduction)
                count+=3
                c=cv2.waitKey(1)
                if c == ord('q') :
                    video.release()
                    return
            video.release()

    video.release()
    return
def mahjong_main(cap,m,dst,ton_player,field_points,cM,size,player_points,min_size=(540,960,3),save_time=None,round_wind=0,honba=0,save_movie=None,effect=None):
    #表示倍率
    reduction=size[0]/min_size[0]
    
    #目印用表示
    img=draw.draw_rect(field_points,size,reduction=reduction)
    img=draw.draw_ura_rect(field_points,size,img,reduction=reduction)
    img=draw.draw_kaze(field_points,ton_player,img=img,reduction=reduction)
    # img=draw.draw_honba(field_points,ton_player,round_wind,honba,img=img,reduction=reduction)
    
    #投影
    sM=show_img(img,m,field_points,dst=dst,reduction=reduction)

    #サイコロ用変数
    dice_count=0
    dice_rand=random.randint(20,30)
    dice_number=[0,0]

    while(1):
        ret, im = cap.read()
        im=trans.transform_camera(im,M=cM)
        if save_movie is not None:
            save_movie.write(cv2.resize(im,(1920,1080)))
        if effect is not None:
            effect.write(cv2.resize(img,(1920,1080)))
        cv2.imshow('Camera',cv2.resize(im,(1920,1080)))

        #サイコロの表示
        if dice_count<2*dice_rand and dice_count%2==0:
            dice_img,dice_number=draw.draw_dice(field_points,size,img,reduction,dice_number)
            show_img(dice_img,m,field_points,dst=dst,reduction=reduction,M=sM)
        dice_count+=1

        c=cv2.waitKey(1)
        if c  == ord('q'):
            break
        elif c == ord('p'):
            #終了
            return 0,save_time,False
        
    isRead=True
    while(isRead):
        reduction=size[0]/min_size[0]
        win_player=read_trigger(cap,field_points,size,cM,ton_player,m,round_wind,honba,dst=dst,reduction=reduction,save_movie=save_movie,effect=effect)
        st_time=time.time()
        if win_player==-1:
            return 0,save_time,True
        elif win_player==-2:
            return 2,save_time,True
        draw_flag=False
        
        read_size=(1080,1920,3)
        while(isRead):
            if draw_flag:
                img=draw.draw_hand(field_points,hand_classes,hand_boxes,win_player,size,reduction=size[0]/read_size[0])
                img=draw.draw_dora(field_points,dora_classes,dora_boxes,win_player,size,img,reduction=size[0]/read_size[0])
                img=draw.draw_naki(field_points,naki_classes,naki_boxes,win_player,size,img,reduction=size[0]/read_size[0])
                img=draw.draw_wintile(field_points,win_class,win_box,win_player,size,img,reduction=size[0]/read_size[0])
            else:
                img=None

            wait_no_wintile(field_points,win_player,size,dst,cap,cM,m,img,reduction=size[0]/read_size[0],save_movie=save_movie,effect=effect)
            
            #牌配置待機
            win_class,win_box,lose_player=read_wintile(field_points,win_player,size,cap,cM,ton_player,m,dst=dst,is_eval_draw=True,reduction=size[0]/read_size[0],save_movie=save_movie,effect=effect)
            if win_class==-1:
                break
            if win_class==-2:
                if ton_player==win_player:
                    return 0,save_time,True
                else:
                    return 1,save_time,True
            sub_time=time.time()
            

            time.sleep(1)
            def_img=draw.draw_player_rect(field_points,win_player,size,reduction=reduction)
            def_img=draw.draw_kaze(field_points,ton_player,img=def_img,reduction=reduction)
            show_img(def_img,m,field_points,M=sM,reduction=reduction)
            for i in range(500):
                cv2.waitKey(1)
            ret, im = cap.read()
            im=trans.transform_camera(im,M=cM)
            # 点数計算
            hai_img=get.get_hand(field_points,win_player,im,size)
            hand_classes,hand_scores,hand_boxes=eval.hand_eval(hai_img,0.3)
            hai_img=get.get_dora(field_points,win_player,im,size)
            dora_classes,dora_scores,dora_boxes=eval.dora_eval(hai_img,0.6)
            hai_img=get.get_naki(field_points,win_player,im,size)
            naki_classes,naki_scores,naki_boxes=eval.naki_eval(hai_img,0.6)

            
            print('agari',MAHJONG_CLASSES[win_class[0]])
            print('hand',hand_scores)
            for hand_class in hand_classes:
                print(MAHJONG_CLASSES[hand_class])
            print('dora',dora_scores)
            for dora_class in dora_classes:
                print(MAHJONG_CLASSES[dora_class])
            print('naki',naki_scores)
            for naki_class in naki_classes:
                print(MAHJONG_CLASSES[naki_class])
            is_tsumo=win_player==lose_player
            result=mahjong_calculation.mahjong_auto(hand_classes,naki_classes,naki_boxes,dora_classes,dora_boxes,win_class,win_box,get_wind(win_player,ton_player),round_wind=round_wind,honba=honba,is_tsumo=is_tsumo)
            
            draw_flag=True
            if result==-1:
                music.play_music("./music/ビープ音1.mp3")
                continue
            print(result)
            print(result.yaku)
            if type(result.han) is not int:
                music.play_music("./music/ビープ音1.mp3")
                continue
            
            en_time=time.time()
            save_time=pd.concat([save_time,pd.Series([en_time-st_time,en_time-sub_time])],axis=1)
            agari=get.get_agari(result)
            #演出表示
            if agari!=-1:
                draw_movie(field_points,size,m,cap,win_player,cM,agari,dst=dst,save_movie=save_movie,effect=effect)
            
            #結果の表示
            music.play_music(POINT_SE[agari+1])
            reduction=size[0]/read_size[0]
            img=draw.draw_result(result,field_points,win_player,size,reduction=reduction)
            img=draw.draw_hand(field_points,hand_classes,hand_boxes,win_player,size,img,reduction=reduction)
            img=draw.draw_dora(field_points,dora_classes,dora_boxes,win_player,size,img,reduction=reduction)
            img=draw.draw_naki(field_points,naki_classes,naki_boxes,win_player,size,img,reduction=reduction)
            img=draw.draw_wintile(field_points,win_class,win_box,win_player,size,img,reduction=reduction)
            img=draw.draw_kaze(field_points,ton_player,img=img,reduction=reduction)
            _=show_img(img,m,field_points,dst=dst,reduction=reduction)
            cv2.waitKey(1)
            
            if agari != -1:
                img=draw.draw_agari(agari,field_points,win_player,size,reduction=reduction)
                img=draw.draw_result(result,field_points,win_player,size,img,reduction=reduction)
                img=draw.draw_hand(field_points,hand_classes,hand_boxes,win_player,size,img,reduction=reduction)
                img=draw.draw_dora(field_points,dora_classes,dora_boxes,win_player,size,img,reduction=reduction)
                img=draw.draw_naki(field_points,naki_classes,naki_boxes,win_player,size,img,reduction=reduction)
                img=draw.draw_wintile(field_points,win_class,win_box,win_player,size,img,reduction=reduction)
                img=draw.draw_kaze(field_points,ton_player,img=img,reduction=reduction)
                time.sleep(1)
                if agari==4:
                    time.sleep(1)
                music.play_music(AGARI_SE[int(agari//3)])
                
                _=show_img(img,m,field_points,dst=dst,reduction=reduction)
            
            while(1):
                ret, im = cap.read()
                im=trans.transform_camera(im,M=cM)
                if save_movie is not None:
                    save_movie.write(cv2.resize(im,(1920,1080)))
                if effect is not None:
                    effect.write(cv2.resize(img,(1920,1080)))
                cv2.imshow("Camera", cv2.resize(im,(1920,1080)))
                c=cv2.waitKey(1)

                #終了
                if c == ord('q'):
                    isRead=False
                    break
                #点数間違い
                if c == ord('p'):
                    break
    
    #点数変更
    if is_tsumo:
        player_points[win_player-1]+=result.cost['main']+result.cost['additional']*2
        #減点
        for i in range(4):
            if i+1!=win_player:
                if i+1==ton_player:
                    player_points[i]-=result.cost['main']
                else:
                    player_points[i]-=result.cost['additional']
    else:
        player_points[win_player-1]+=result.cost['main']
        player_points[lose_player-1]-=result.cost['main']
    if ton_player==win_player:
        return 0,save_time,True
    else:
        return 1,save_time,True

def main():
    now=datetime.datetime.now()
    time_df=pd.DataFrame()
    
    #カメラの設定
    m=get_monitors()[1]
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
    cap.set(cv2.CAP_PROP_FOCUS,0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,3840)

    #動画保存用設定
    effect=None
    effect=save_video(cap,"./save_movie/all_effect.mp4")
    save_movie=None
    save_movie=save_video(cap,"./save_movie/all_movie.mp4")

    #モデルを読み込んでおく
    _ = eval.trigger_eval(np.zeros([100,100,3],dtype=np.uint8))
    _,_,_ = eval.win_eval(np.zeros([100,100,3],dtype=np.uint8),0.9)
    
    # カメラ調節
    while(cap.isOpened()):
        ret, im = cap.read()
        if ret:
            def_points=area.get_green(im)
            if len(def_points) > 0:
                #判定領域の表示
                new_im=im.copy()
                cv2.polylines(new_im,[def_points],True,(0, 0, 255),4)
                cv2.imshow("Camera", cv2.resize(new_im,(1920,1080)))

                if save_movie is not None:
                    save_movie.write(cv2.resize(im,(1920,1080)))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    im_middle=[im.shape[0]//2,im.shape[1]//2]
    field_points=[[im_middle[1]-im_middle[0]+10,10],[im_middle[1]+im_middle[0]-10,im.shape[0]-10]]
    
    #カメラ
    def_points=area.mask_sort(def_points)
    print("field",field_points)
    print("def",def_points)
    save_im,cM=trans.transform_camera(im,field_points,src=def_points)
    save_im[:field_points[0][1], :] = 0  # 上部
    save_im[field_points[1][1]:, :] = 0  # 下部
    save_im[:, :field_points[0][0]] = 0  # 左側
    save_im[:, field_points[1][0]:] = 0  # 右側
    
    #投影設定
    size=im.shape
    middle=[size[1]//2,size[0]//2]
    
    dst=np.float32([np.array([middle[0]-size[1]//10,middle[1]//30]),np.array([middle[0]+size[1]//3,middle[1]//30]),
                    np.array([middle[0]-size[1]//10,middle[1]*19//10]),np.array([middle[0]+size[1]//3,middle[1]*19//10])])
    
    reduction=size[0]/size[0]
    min_field=([int(field_points[0][0]//reduction),int(field_points[0][1]//reduction)],[int(field_points[1][0]//reduction),int(field_points[1][1]//reduction)])
    cv2.destroyAllWindows()
    #領域の投影
    img=np.zeros(size,np.uint8)
    cv2.rectangle(img, min_field[0], min_field[1], (255, 0, 0),-1)
    _=show_img(img,m,min_field,dst=dst)
    cv2.waitKey(500)
    a=50
    def_points=[[field_points[0][0]+a,field_points[0][1]+a]
                ,[field_points[1][0]-a,field_points[1][1]-a]] 
    
    #投影調節
    sum=30
    isBreak=False
    while(cap.isOpened()):
        ret, im = cap.read()
        im=trans.transform_camera(im,M=cM)
        if save_movie is not None:
            save_movie.write(cv2.resize(im,(1920,1080)))
        im[:field_points[0][1], :] = 0  # 上部
        im[field_points[1][1]:, :] = 0  # 下部
        im[:, :field_points[0][0]] = 0  # 左側
        im[:, field_points[1][0]:] = 0  # 右側
        mask=area.get_rect(im,save_im)
        cv2.rectangle(im,def_points[0],def_points[1],(0,255,0),3)
        if len(mask)>0:
            cv2.polylines(im,[mask],True,(0, 0, 255),3)
            dst,isBreak=area.get_dst(field_points,mask,dst,max=size[0])
        cv2.imshow("mask", cv2.resize(im,(1920,1080)))
        img=np.zeros(size,np.uint8)
        cv2.rectangle(img, min_field[0], min_field[1], (255, 0, 0),-1)
        if effect is not None:
            effect.write(cv2.resize(img,(1920,1080)))
        _=show_img(img,m,min_field,dst=dst)
        
        if cv2.waitKey(500) & 0xFF == ord('q') or isBreak:
            break

    cv2.destroyAllWindows()
    
    #初期設定
    size=im.shape
    round_wind=0
    isContinue=True
    player_points=[25000,25000,25000,25000]

    #ゲーム開始
    while(isContinue):
        while(ton_player<=4 and isContinue):
            #局の開始
            win_result,time_df,isContinue=mahjong_main(cap,m,dst,ton_player,field_points,cM,size,player_points,save_time=time_df,round_wind=round_wind,honba=honba,save_movie=save_movie,effect=effect)
            
            if win_result>0:
                ton_player+=1
            honba+=1
            if win_result==1:
                honba=0
        
        ton_player=1
        round_wind+=1
        if round_wind>3:
            round_wind=0
        print(round_wind)

        
    cap.release()
    music.play_music("./music/成功音.mp3")
    c=cv2.waitKey()
    effect.release()


    return



if __name__ == '__main__':
    # camera.riichi_camera()
    main()
