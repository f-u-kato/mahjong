import cv2
from screeninfo import get_monitors

import numpy as np
import random
import time
import datetime
import pandas as pd
import os
import glob

import draw_img as draw
import mahjong_eval as eval
import get_area as area
import get_img as get
import data.mahjong.calculation as mahjong_calculation
import play_music as music



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
TRIGGER_SE = [r'.\music\効果音1.mp3']
LOAD_SE=r'.\music\loading.mp3'
AGARI_SE=[r'.\music\和太鼓でドン.mp3',r'.\music\和太鼓でドドン.mp3']
RYOUKYOKU_SE=r'.\music\しょげる.mp3'
POINT_SE=[r'.\music\平手打ち1.mp3',r'.\music\剣で斬る2.mp3',r'.\music\剣で斬る1.mp3',r'.\music\剣で斬る3.mp3',r'.\music\剣で斬る4.mp3',r'.\music\剣で斬る6.mp3']
AGARI_IMAGES = ['./material/points/mangan.png','./material/points/haneman.png','./material/points/baiman.png','./material/points/3bai.png','./material/points/yakuman.png']
BACK_MOVIES=glob.glob("./material/back/*")
AGARI_VIDEOS=['./material/満貫.mp4','./material/跳満.mp4','./material/倍満.mp4','./material/三倍満.mov']
YAKUMAN_VIDEOS=['./material/役満1.mp4','./material/役満2.mov','./material/役満3.mp4']



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

#カメラの補正
def transform_camera(path,dst=None,src=None,M=None):
    img=path.copy()
    height,width,_=img.shape
    if M is None:
        p1=dst[0][0]
        p2=dst[0][1]
        p3=dst[1][0]
        p4=dst[1][1]
        #2160,3840,3
        dst=np.float32([np.array([p1,p2]),np.array([p3,p2]),np.array([p1,p4]),np.array([p3,p4])])
        if src is None:
            src=[np.array([p1,p2]),np.array([p3,p2+height//20]),np.array([p1,p4]),np.array([p3-width//60,p4-height//60])]
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
def transform_img(path,dst=None,src=None,M=None):
    img=path.copy()
    
    height,width,_=img.shape
    p1=dst[0][0]
    p2=dst[0][1]
    p3=dst[1][0]
    p4=dst[1][1]
    img=img[p2:p4][:]
    if M is None:
        field_height=p4-p2
        src=np.float32([np.array([p1,p2]),np.array([p3,p2]),np.array([p1,p4]),np.array([p3,p4])])
        p2=0
        #2160,3840,3
        p4=height
        
        field_width=p3-p1
        dst=np.float32([np.array([p1+field_width//5,p2+field_height//20]),np.array([p3+field_width//3-field_width//30,p2+field_height//15]),
                        np.array([p1+field_width//4-field_width//30,p4+field_height//15]),np.array([p3+field_width//3-field_width//30,p4+field_height//20])])
        
        # 変換行列
        M = cv2.getPerspectiveTransform(src, dst)
    
        # 射影変換・透視変換する
        output = cv2.warpPerspective(img, M,(width, height))
        return output,M
    else:
        output = cv2.warpPerspective(img, M,(int(width), int(height)))
        return output



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


def show_img(img,size_data,field_points,M=None,reduction=1):
    draw_points=([int(field_points[0][0]//reduction),int(field_points[0][1]//reduction)],[int(field_points[1][0]//reduction),int(field_points[1][1]//reduction)])

    # ウィンドウを作成する
    cv2.namedWindow("Projector Output", cv2.WINDOW_NORMAL)

    # 画像を表示する
    if M is None:
        new_im,M=transform_img(img,dst=draw_points)
        cv2.imshow("Projector Output", new_im)
        cv2.moveWindow('Projector Output',size_data.x,size_data.y)
        # プロジェクターに接続する
        cv2.setWindowProperty("Projector Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow("Projector Output", size_data.width, size_data.height)
        return M
    else:
        cv2.imshow("Projector Output", transform_img(img,dst=draw_points,M=M))
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

def read_trigger(cap, field_points, size, cM, ton_player, m, round_wind, honba,reduction=1):
    img = draw.draw_rect(field_points,size,reduction=reduction)
    img = draw.draw_kaze(field_points,ton_player,img=img,reduction=reduction)
    sM = show_img(img,m,field_points,reduction=reduction)
    cv2.waitKey(1)
    #音楽再生
    rand=random.randint(0,len(PLAY_BGM)-1)
    music.loop_music(PLAY_BGM[rand])
    count = 0
    save=save_video(cap,"read_trigger.mp4")
    while(cap.isOpened()):
        ret, im = cap.read()
        im = transform_camera(im,M=cM)
        save.write(cv2.resize(im,(1920,1080)))
        im = draw.draw_rect2(field_points,size,im)
        cv2.imshow("Camera", cv2.resize(im,(1920,1080)))
        
        win_player = check_tile(field_points, im, size)
        if win_player > 0:
            print('check')
            break
        img = draw.draw_rect(field_points,size,reduction=reduction)
        img = draw.draw_kaze(field_points,ton_player,img=img,reduction=reduction)
        # img = draw.draw_honba(field_points,ton_player,round_wind,honba,img=img,reduction=reduction)
        show_img(img,m,field_points,sM,reduction=reduction)
        c = cv2.waitKey(1)
        if c == ord('q'):
            save.release()
            music.stop_music()
            music.play_music(RYOUKYOKU_SE)
            return -1
        elif c == ord('p'):
            save.release()
            music.stop_music()
            music.play_music(RYOUKYOKU_SE)
            return -2
    save.release()
    return win_player

def wait_no_wintile(field_points,win_player,size,sM,cap,cM,m,im=None,reduction=1):
    img=draw.draw_player_rect(field_points,win_player,size,first=True,img=im,reduction=reduction)
    show_img(img,m,field_points,sM,reduction=reduction)
    while(cap.isOpened()):
        ret, im = cap.read()
        im=transform_camera(im,M=cM)
        if not in_hai(get.get_wintile(field_points,win_player,im,size)):
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            music.stop_music()
            break

#上がり牌のチェック
def read_wintile(field_points,win_player,size,cap,cM,sM,ton_player,m,is_eval_draw=True,reduction=1):
    music.stop_music()
    music.play_music(TRIGGER_SE[random.randint(0,len(TRIGGER_SE)-1)])

    def_img=draw.draw_player_rect(field_points,win_player,size,reduction=reduction)
    def_img=draw.draw_kaze(field_points,ton_player,img=def_img,reduction=reduction)
    # for i in range(4):
    #     img=draw.draw_player_rect(field_points,i+1,size,img)
    img=def_img.copy()
    while(cap.isOpened()):
        ret, im = cap.read()
        show_img(img,m,field_points,sM,reduction=reduction)
        cv2.waitKey(1)
        im=transform_camera(im,M=cM)
        
        new_im=im.copy()
        new_im=draw.draw_player_rect2(field_points,win_player,size,new_im)
        cv2.imshow("Camera", cv2.resize(new_im,(1920,1080)))
        hai_img=get.get_wintile(field_points,win_player,im,size)
        if in_hai(hai_img,0.9):
            win_class,win_score,win_box=eval.win_eval(hai_img,0.8)
            if len(win_class)>0 and win_class!=37:
                print('set ok')
                music.play_music(LOAD_SE)
                break
        #結果の表示
        if is_eval_draw:
            hai_img=get.get_hand(field_points,win_player,im,size)
            hand_classes,hand_scores,hand_boxes=eval.hand_eval(hai_img,0.5)
            hai_img=get.get_dora(field_points,win_player,im,size)
            dora_classes,dora_scores,dora_boxes=eval.dora_eval(hai_img,0.5)
            hai_img=get.get_naki(field_points,win_player,im,size)
            naki_classes,naki_scores,naki_boxes=eval.naki_eval(hai_img,0.6)
            img=draw.draw_hand(field_points,hand_classes,hand_boxes,win_player,size,def_img.copy(),reduction=reduction)
            img=draw.draw_dora(field_points,dora_classes,dora_boxes,win_player,size,img,reduction=reduction)
            img=draw.draw_naki(field_points,naki_classes,naki_boxes,win_player,size,img,reduction=reduction)
            


        c=cv2.waitKey(1)
        if c == ord('q'):
            return -1,-1
    return win_class,win_box



def draw_movie(field_points,size,m,cap,win_player,cM,agari):
    min_size=(540,960,3)
    reduction=size[0]/min_size[0]
    min_field=([int(field_points[0][0]//reduction),int(field_points[0][1]//reduction)],[int(field_points[1][0]//reduction),int(field_points[1][1]//reduction)])
    img=np.zeros(min_size,np.uint8)
    min_sM=show_img(cv2.resize(img,(min_size[1],min_size[0])),m,min_field)
    if agari<4:
        video=AGARI_VIDEOS[agari]
        video=cv2.VideoCapture(AGARI_VIDEOS[agari])
        save=save_video(cap,"result.mp4")
        count=0
        while(cap.isOpened()):
            ret, im = cap.read()
            im=transform_camera(im,M=cM)
            save.write( cv2.resize(im,(1920,1080)))
            cv2.imshow("Camera", cv2.resize(im,(1920,1080)))
            img=np.zeros(min_size,np.uint8)
            frame=draw.back_place(video,img,min_field,win_player,count)
            if frame is None:
                video.release()
                return
            show_img(frame,m,min_field,M=min_sM)
            c=cv2.waitKey(1)
            count+=2
            if c == ord('q') :
                video.release()
                return
    else:
        save=save_video(cap,"result.mp4")
        for i in range(len(YAKUMAN_VIDEOS)):
            video=YAKUMAN_VIDEOS[i]
            video=cv2.VideoCapture(video)
            if i==2:
                music.play_music("./music/和風ジングル.mp3")
            count=0
            while(cap.isOpened()):
                ret, im = cap.read()
                im=transform_camera(im,M=cM)
                save.write( cv2.resize(im,(1920,1080)))
                cv2.imshow("Camera", cv2.resize(im,(800,450)))
                img=np.zeros(min_size,np.uint8)
                frame=draw.back_place(video,img,min_field,win_player,count)
                if frame is None:
                    break
                show_img(frame,m,min_field,M=min_sM)
                count+=3
                c=cv2.waitKey(1)
                if c == ord('q') :
                    video.release()
                    return
            video.release()

    video.release()
    save.release()
    return
def mahjong_main(cap,m,ton_player,field_points,cM,size,save_time=None,round_wind=0,honba=0):
    color = (0, 255, 0)
    min_size=(720,1280,3)
    reduction=size[0]/min_size[0]
    #かぶらないように調節
    img=draw.draw_rect(field_points,size,reduction=reduction)
    img=draw.draw_ura_rect(field_points,size,img,reduction=reduction)
    img=draw.draw_kaze(field_points,ton_player,img=img,reduction=reduction)
    # img=draw.draw_honba(field_points,ton_player,round_wind,honba,img=img,reduction=reduction)
    sM=show_img(img,m,field_points,reduction=reduction)
    while(1):
        ret, im = cap.read()
        im=transform_camera(im,M=cM)
        cv2.imshow('Camera',cv2.resize(im,(1920,1080)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    isRead=True
    while(isRead):
        win_player=read_trigger(cap,field_points,size,cM,ton_player,m,round_wind,honba,reduction=reduction)
        st_time=time.time()
        if win_player==-1:
            return 0,save_time
        elif win_player==-2:
            return 2,save_time
        draw_flag=False
        while(isRead):
            if draw_flag:
                img=draw.draw_hand(field_points,hand_classes,hand_boxes,win_player,size,reduction=reduction)
                img=draw.draw_dora(field_points,dora_classes,dora_boxes,win_player,size,img,reduction=reduction)
                img=draw.draw_naki(field_points,naki_classes,naki_boxes,win_player,size,img,reduction=reduction)
                img=draw.draw_wintile(field_points,win_class,win_box,win_player,size,img,reduction=reduction)
            else:
                img=None
            wait_no_wintile(field_points,win_player,size,sM,cap,cM,m,img,reduction=reduction)
            
            #牌配置待機
            win_class,win_box=read_wintile(field_points,win_player,size,cap,cM,sM,ton_player,m,is_eval_draw=True,reduction=reduction)
            if win_class==-1:
                break
            sub_time=time.time()
            

            time.sleep(1)
            def_img=draw.draw_player_rect(field_points,win_player,size,reduction=reduction)
            def_img=draw.draw_kaze(field_points,ton_player,img=def_img,reduction=reduction)
            show_img(def_img,m,field_points,sM,reduction=reduction)
            cv2.waitKey(500)
            ret, im = cap.read()
            im=transform_camera(im,M=cM)
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
            result=mahjong_calculation.mahjong_auto(hand_classes,naki_classes,naki_boxes,dora_classes,dora_boxes,win_class,win_box,get_wind(win_player,ton_player),round_wind=round_wind,honba=honba)
            # result=mahjong_calculation.haneman_result()
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
                draw_movie(field_points,size,m,cap,win_player,cM,agari)
            
            #結果の表示
            music.play_music(POINT_SE[agari+1])
            img=draw.draw_result(result,field_points,win_player,size,reduction=reduction)
            img=draw.draw_hand(field_points,hand_classes,hand_boxes,win_player,size,img,reduction=reduction)
            img=draw.draw_dora(field_points,dora_classes,dora_boxes,win_player,size,img,reduction=reduction)
            img=draw.draw_naki(field_points,naki_classes,naki_boxes,win_player,size,img,reduction=reduction)
            img=draw.draw_wintile(field_points,win_class,win_box,win_player,size,img,reduction=reduction)
            img=draw.draw_kaze(field_points,ton_player,img=img,reduction=reduction)
            show_img(img,m,field_points,sM,reduction=reduction)
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
                show_img(img,m,field_points,sM,reduction=reduction)
            
            while(1):
                ret, im = cap.read()
                im=transform_camera(im,M=cM)
                cv2.imshow("Camera", cv2.resize(im,(1920,1080)))
                c=cv2.waitKey(1)

                if c == ord('q'):
                    isRead=False
                    break
                if c == ord('p'):
                    break
            cv2.imwrite("./result.png",cv2.resize(im,(1920,1080)))

    
    if ton_player==win_player:
        return 0,save_time
    else:
        return 1,save_time

def main():
    now=datetime.datetime.now()
    time_df=pd.DataFrame()
    m=get_monitors()[1]
    ton_player=1
    honba=0
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
    cap.set(cv2.CAP_PROP_FOCUS,0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,3840)
    min_size=(720,1280,3)
    # カメラ調節
    _ = eval.trigger_eval(np.zeros([100,100,3],dtype=np.uint8))
    while(cap.isOpened()):
        ret, im = cap.read()
        field_points,_=area.get_green(im)
        color = (0, 255, 0)
        # im,_=transform_camera(im,dst=field_points)

        if len(field_points) > 0:
            new_im=im.copy()
            cv2.rectangle(new_im, field_points[0], field_points[1], color,3)
            cv2.imshow("Camera", cv2.resize(new_im,(1920,1080)))
            size=im.shape
            reduction=size[0]/min_size[0]
            min_field=([int(field_points[0][0]//reduction),int(field_points[0][1]//reduction)],[int(field_points[1][0]//reduction),int(field_points[1][1]//reduction)])
            img=np.zeros(min_size,np.uint8)
            cv2.rectangle(img, min_field[0], min_field[1], (255, 255, 255),1)
            _=show_img(img,m,min_field)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    #-卓領域検出
    while(cap.isOpened()):
        ret, im = cap.read()
        field_points,def_points=area.get_green(im)
        if len(field_points) > 0:
            break
    #カメラ
    def_points=[def_points[0],def_points[3],def_points[1],def_points[2]]
    _,cM=transform_camera(im,dst=field_points,src=def_points)

    #投影
    size=im.shape
    round_wind=0
    while(1):
        while(ton_player<=4):
            win_result,time_df=mahjong_main(cap,m,ton_player,field_points,cM,size,save_time=time_df,round_wind=round_wind,honba=honba)
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
    time_df.to_csv(str(now.date())+'-'+str(now.hour)+'-'+str(now.minute)+'.csv')


    return


def camera():
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
    while(1):
        ret, im = cap.read()
        im=transform_camera(im,M=cM)
        cv2.imshow('Camera',cv2.resize(im,(1920,1080)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.imwrite("./all_field.png",im)
    win_player=1
    hai_img=get.get_hand(field_points,im,win_player,size)
    cv2.imwrite("./hand.png",hai_img)
    hai_img=get.get_dora(field_points,im,win_player,size)
    cv2.imwrite("./dora.png",hai_img)
    hai_img=get.get_naki(field_points,im,win_player,size)
    cv2.imwrite("./naki.png",hai_img)
    hai_img=get.get_wintile(field_points,im,win_player,size)
    cv2.imwrite("./win.png",hai_img)
    

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
        
        
def trigger_camera():
    m=get_monitors()[1]
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
    cap.set(cv2.CAP_PROP_FOCUS,0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,3840)
    # カメラ調節
    while(cap.isOpened()):
        ret, im = cap.read()
        field_points,_=area.get_green(im)
        color = (0, 255, 0)
        # im,_=transform_camera(im,dst=field_points)

        if len(field_points) > 0:
            new_im=im.copy()
            cv2.rectangle(new_im, field_points[0], field_points[1], color,3)
            cv2.imshow("Camera", cv2.resize(new_im,(1920,1080)))
            size=im.shape
            img=np.zeros(size,np.uint8)
            cv2.rectangle(img, field_points[0], field_points[1], (255, 255, 255),3)
            _=show_img(img,m,field_points)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #-卓領域検出
    while(cap.isOpened()):
        ret, im = cap.read()
        field_points,def_points=area.get_green(im)
        if len(field_points) > 0:
            break
    def_points=[def_points[0],def_points[3],def_points[1],def_points[2]]
    _,cM=transform_camera(im,dst=field_points,src=def_points)
    img=draw.draw_rect(field_points,size)
    sM=show_img(img,m,field_points)
    os.makedirs("./trigger/0",exist_ok=True)
    os.makedirs("./trigger/1",exist_ok=True)
    count=0
    # cv2.waitKey()
    # print("start")
    # while(1):
    #     ret, im = cap.read()
    #     im=transform_camera(im,M=cM)
    #     cv2.imshow('Camera',cv2.resize(im,(1920,1080)))
    #     c=cv2.waitKey(1)
        
    #     if count%10==0:
    #         print(count)
    #         get_img=get.get_trigger(field_points,2,im,size)
    #         cv2.imwrite(f"./trigger/0/{count}.png",get_img)
    #     if c == ord('q'):
    #         print("end_ura")
    #         break
    #     count+=1
    cv2.waitKey()
    print("start")
    count=2921
    while(1):
        ret, im = cap.read()
        im=transform_camera(im,M=cM)
        cv2.imshow('Camera',cv2.resize(im,(1920,1080)))
        c=cv2.waitKey(1)
        
        if count%10==0:
            print(count)
            get_img=get.get_trigger(field_points,2,im,size)
            cv2.imwrite(f"./trigger/1/{count}.png",get_img)
        if c == ord('q'):
            print("end")
            break
        count+=1
if __name__ == '__main__':
    # music.loop_music(path)
    # path=r".\data\mahjong\sample\hai.png"
    # img=cv2.imread(path)
    # cv2.imshow('',img)
    # c=cv2.waitKey()
    
    #hai_eval(img)
    # transform_img(img)
    main()
    # camera()
    # hand_camera()
    # trigger_camera()
