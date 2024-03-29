import cv2
from screeninfo import get_monitors
import tkinter as tk
import customtkinter as ctk
from CTkSpinbox import *

import numpy as np
import random
import time
import datetime
import pandas as pd
import os
import glob
import pygame

import src.out_func.draw_img as draw
import src.eval.mahjong_eval as eval
import src.get_func.get_area as area
import src.get_func.get_img as get
import src.eval.calculation as mahjong_calculation
import src.out_func.play_music as music
import src.out_func.transform_video as trans
# import src.out_func.camera as camera

from concurrent.futures import ProcessPoolExecutor


# 結果出力用
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
TENPAI_SE = ["./music/no_ten.mp3","./music/tenpai.mp3"]


MAHJONG_CLASSES = ("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                   "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                   "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
                   "ton", "nan", "sha", "pe",
                   "haku", "hatsu", "chun",
                   "aka_5m", "aka_5p", "aka_5s",
                   "ura")
MAHJONG_CLASSES_NUMBER = {"1m": "1", "2m": "2", "3m": "3", "4m": "4", "5m": "5", "6m": "6", "7m": "7", "8m": "8", "9m": "9",
                          "1p": "1", "2p": "2", "3p": "3", "4p": "4", "5p": "5", "6p": "6", "7p": "7", "8p": "8", "9p": "9",
                          "1s": "1", "2s": "2", "3s": "3", "4s": "4", "5s": "5", "6s": "6", "7s": "7", "8s": "8", "9s": "9",
                          "ton": "1", "nan": "2", "sha": "3", "pe": "4",
                          "haku": "5", "hatsu": "6", "chun": "7",
                          "aka_5m": "0", "aka_5p": "0", "aka_5s": "0",
                          "ura": "-1"}


def get_wind(player_num, ton_num):
    num = player_num-ton_num
    if num < 0:
        num += 4
    return num

# トリガーの判定


def check_tile(field_points, im, size=(2160, 3840, 3), threshold=0.8):
    images = []
    for i in range(4):
        img = get.get_trigger(field_points, i, im)
        images.append(img)

    classes = eval.multi_trigger_eval(images)
    for i, class_num in enumerate(classes):
        if class_num == 0:
            return i
    return -1

# 牌があるかの判定
def in_hai(img, a=0.9):
    height, width, channels = img.shape[:3]
    hsvLower = np.array([30, 50, 40])    # 抽出する色の下限(HSV)
    hsvUpper = np.array([90, 255, 255])    # 抽出する色の上限(HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 画像をHSVに変換
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSVからマスクを作成
    if height*width*a < np.sum(hsv_mask//255) or height*width*0.1 > np.sum(hsv_mask//255):
        return False
    else:
        return True


def show_img(img, size_data=None, field_points=None, dst=None, M=None, reduction=1):

    # ウィンドウを作成する
    cv2.namedWindow("Projector Output", cv2.WINDOW_NORMAL)

    # 画像を表示する
    if M is None:
        new_dst = dst.copy()
        new_dst /= reduction
        draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                       [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
        new_im, M = trans.transform_img(img, dst=new_dst, field_points=draw_points)
        cv2.imshow("Projector Output", new_im)
        cv2.moveWindow('Projector Output', size_data.x, size_data.y)
        # プロジェクターに接続する
        cv2.setWindowProperty("Projector Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow("Projector Output", size_data.width, size_data.height)
        return M
    else:
        cv2.imshow("Projector Output", trans.transform_img(img, M=M))
        cv2.moveWindow('Projector Output', size_data.x, size_data.y)
        # プロジェクターに接続する
        cv2.setWindowProperty("Projector Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow("Projector Output", size_data.width, size_data.height)


def setting_window():
    app = ctk.CTk()
    app.geometry("800x500")
    app.title('麻雀の設定')

    # スコア設定の下の部分に符計算とツモ損の設定を追加する関数
    def add_sanma_options():
        # 既にオプションが存在していれば何もしない
        if hasattr(app, 'sanma_option_frame'):
            return
        
        # トグルボタンの値を保持する変数
        app.fu_calculation = tk.BooleanVar(value=False)
        app.tsumo_loss = tk.BooleanVar(value=False)
        
        # トグルボタンを配置するフレームを作成
        app.sanma_option_frame = ctk.CTkFrame(app)
        app.sanma_option_frame.pack(side="top", anchor="n", pady=10)
        
        # 「符計算なし」のトグルボタンを作成・配置
        ctk.CTkSwitch(app.sanma_option_frame, text="符計算なし", variable=app.fu_calculation, onvalue=True, offvalue=False).pack(side="left", padx=10)
        # 「ツモ損あり」のトグルボタンを作成・配置
        ctk.CTkSwitch(app.sanma_option_frame, text="ツモ損なし", variable=app.tsumo_loss, onvalue=True, offvalue=False).pack(side="left", padx=10)

    # オプションを削除する関数
    def remove_sanma_options():
        if hasattr(app, 'sanma_option_frame'):
            app.sanma_option_frame.pack_forget()  # フレームをパックから解除
            app.sanma_option_frame.destroy()      # フレームを破棄
            del app.sanma_option_frame            # オブジェクトの属性を削除

    # 東南モードが変更された時に呼ばれる関数
    def on_mode_change():
        # is_sanmaがTrueの場合、追加オプションを表示
        if mode_type.get() > 1:
            add_sanma_options()
        else:
            # 三人麻雀でない場合、オプションを削除する
            remove_sanma_options()


    # スピンボックスの有効/無効を切り替える関数
    def toggle_spinbox(spinbox,is_check):
        if is_check.get() == 1:  # チェックボックスがチェックされていない場合
            spinbox.configure(text_color='white')  # spinboxを無効にする
        else:  # チェックボックスがチェックされている場合
            spinbox.configure(text_color='black')  # spinboxを有効にする

    # ラジオボタン
    mode_type = tk.IntVar(value=0)
    radio_frame = ctk.CTkFrame(app)
    radio_frame.pack(side="top", anchor="n", pady=20) 
    # モード変更時のコマンドを追加してラジオボタンを作成
    for text, value in [("四人東", 0), ("四人南", 1), ("三人東", 2), ("三人南", 3)]:
        ctk.CTkRadioButton(radio_frame, text=text, variable=mode_type, value=value, command=on_mode_change).pack(side="left", padx=10)

    # スコア設定
    score_frame = ctk.CTkFrame(app)
    score_frame.pack(side="top", anchor="n",pady=20)
    ctk.CTkLabel(score_frame, text="終了条件(基準点数)").pack(side="left", padx=10)
    end_score = CTkSpinbox(score_frame,start_value=30000, min_value=0, max_value=100000, scroll_value=100,step_value=100, width=200)
    end_score.pack(side="left")
    is_check=tk.IntVar(value=1)
    ctk.CTkCheckBox(score_frame, text="設定しない", variable=is_check, onvalue=0, offvalue=1, command=lambda:toggle_spinbox(end_score,is_check)).pack(side="left", padx=10)


    # スタートボタンの作成と配置
    ctk.CTkButton(app, text="スタート", command=lambda: app.destroy()).pack(side="right", anchor="n", padx=80, pady=50)

    app.mainloop()

    is_sanma = mode_type.get() > 1
    is_tonpu = mode_type.get() % 2 == 0
    score = end_score.get() if is_check.get() else 0
    sanma_options_result = [app.fu_calculation.get(), app.tsumo_loss.get()] if is_sanma else [None, None]

    return is_sanma, is_tonpu, score, sanma_options_result

def read_trigger(cap, field_points, size, cM, ton_player, m, round_wind, honba, kyotaku, dst, player_points, reduction=1, is_sanma=False):
    # 立直の判定
    isRiichi = [False, False, False, False]
    # 立直判定のカウント
    r_count = [0, 0, 0, 0]
    r_max = 2
    r_video = None
    r_frame = [0, 0, 0, 0]
    # 動画の再生速度
    speed = 1
    # トリガー動画の読み込み
    trigger = cv2.VideoCapture(TRIGGER_VIDEOS[random.randint(0, len(TRIGGER_VIDEOS)-1)])
    # 情報と領域の投影
    img = draw.draw_rect_movie(field_points, trigger, size, img=None, reduction=reduction)
    img = draw.draw_riichi(field_points, img=img, reduction=reduction)
    img = draw.draw_kaze(field_points, ton_player, img=img, reduction=reduction, is_sanma=is_sanma)
    img = draw.draw_honba(field_points, ton_player, round_wind, honba, kyotaku,img=img, reduction=reduction)
    img = draw.draw_player_points(field_points, player_points, img=img, reduction=reduction,is_sanma=is_sanma)
    sM = show_img(img, m, field_points, dst=dst, reduction=reduction)
    cv2.waitKey(1)
    # 音楽再生
    rand = random.randint(0, len(PLAY_BGM)-1)
    music.loop_music(PLAY_BGM[rand])
    count = 0
    # 背景動画の読み込み
    video = cv2.VideoCapture(BACK_MOVIES[rand])

    with ProcessPoolExecutor(max_workers=2) as executor:
        # カメラ映像の取得
        ret, im = cap.read()
        im = trans.transform_camera(im, M=cM)
        # トリガー検出
        trigger_process = executor.submit(check_tile, field_points, im.copy(), size)

        # 立直判定
        riichi_images = []
        for i in range(4-is_sanma):
            riichi_images.append(get.get_riichi(field_points, i, im.copy()))
        riichi_process = executor.submit(eval.multi_riichi_eval, riichi_images)


        while (cap.isOpened()):
            # カメラ映像の取得
            ret, im = cap.read()
            im = trans.transform_camera(im, M=cM)

            # カメラ映像の表示
            out_im = draw.draw_rect2(field_points, size, im)
            out_im = draw.draw_riichi2(field_points, out_im, size)
            cv2.imshow("Camera", cv2.resize(out_im, (1920, 1080)))

            # トリガー検出
            if trigger_process.done():
                win_player = trigger_process.result()
                if win_player > -1:
                    print('check')
                    break
                trigger_process = executor.submit(check_tile, field_points, im.copy(), size)
            

            # 立直判定
            if sum(isRiichi) < len(isRiichi):
                if riichi_process.done():
                    riichi_evals = riichi_process.result()
                    for i, riichi_eval in enumerate(riichi_evals):
                        if riichi_eval == 1 and not isRiichi[i]:
                            r_count[i] += 1
                            # 立直判定が一定数以上の場合，立直を宣言
                            if r_count[i] > r_max:
                                # 初めての立直の場合，立直の音楽を再生
                                if r_video is None:
                                    music.stop_music()
                                    r_video = cv2.VideoCapture(RIICHI_VIDEO)
                                    speed *= 2
                                # リーチ演出の再生
                                music.play_se(RIICHI_SE[0])
                                isRiichi[i] = True
                                # 立直の場合，点数を減らす
                                player_points[i] -= 1000
                            else:
                                isRiichi[i] = False
                        elif not isRiichi[i]:
                            r_count[i] = 0
                    if sum(isRiichi) < 4 - is_sanma:
                        riichi_images = []
                        for i in range(4-is_sanma):
                            riichi_images.append(get.get_riichi(field_points, i, im))
                        riichi_process = executor.submit(eval.multi_riichi_eval, riichi_images)

            # 背景動画の投影
            img = draw.loop_movie(field_points, video, size, ton_player, reduction=reduction, speed=speed)

            # 立直演出
            for i in range(4-is_sanma):
                if isRiichi[i] and r_count[i] > r_max:
                    tmp_img = draw.back_place(r_video, img, field_points, i, time=r_frame[i], reduction=reduction,skelton=True)
                    r_frame[i] += 3
                    if tmp_img is None:
                        if not (-1 in r_count):
                            music.loop_music(RIICHI_BGM[rand])
                        r_count[i] = -1
                    else:
                        img = tmp_img

            # トリガー動画の投影
            img = draw.draw_rect_movie(field_points, trigger, size, img=img, reduction=reduction, is_sanma=is_sanma)
            # 情報の投影
            img = draw.draw_riichi(field_points, img=img, reduction=reduction)
            img = draw.draw_kaze(field_points, ton_player, img=img, reduction=reduction,is_sanma=is_sanma)
            img = draw.draw_honba(field_points, ton_player, round_wind, honba, kyotaku+sum(isRiichi),img=img, reduction=reduction)
            img = draw.draw_player_points(field_points, player_points, img=img, reduction=reduction,is_sanma=is_sanma)
            show_img(img, m, field_points, M=sM, reduction=reduction)
            c = cv2.waitKey(1)
            count += 1
            # 流局
            if c == ord('q') and count > 50:
                music.stop_music()
                music.play_se(RYOUKYOKU_SE)
                win_player = -1
                break
            # リーチのリセット
            elif c == ord('r'):
                for i in range(4-is_sanma):
                    player_points[i] += 1000 * isRiichi[i]
                    isRiichi[i] = False
                r_count = [0, 0, 0, 0]
                speed = 1
                music.stop_music()
                music.loop_music(PLAY_BGM[rand])
                

    return win_player, isRiichi

# 上がり牌配置領域に牌がなくなるまで待機
def wait_no_wintile(field_points, win_player, size, dst, cap, cM, m, im=None, reduction=1, is_sanma=False):
    img = draw.draw_player_rect(field_points, win_player, size, first=True, img=im, reduction=reduction, is_sanma=is_sanma)
    show_img(img, m, field_points, dst=dst, reduction=reduction)
    while (cap.isOpened()):
        ret, im = cap.read()
        im = trans.transform_camera(im, M=cM)

        isHai = False
        for i in range(4-is_sanma):
            if in_hai(get.get_wintile(field_points, i, im, size)):
                isHai = True
        if not isHai:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            music.stop_music()
            break

# 上がり牌のチェック
def read_wintile(field_points, win_player, size, cap, cM, ton_player, m, dst, is_eval_draw=True, reduction=1, is_sanma=False):
    # SEの再生
    music.stop_music()
    music.play_se(TRIGGER_SE[random.randint(0, len(TRIGGER_SE)-1)])
    # 領域の投影
    def_img = draw.draw_player_rect(field_points, win_player, size, reduction=reduction, is_sanma=is_sanma)
    def_img = draw.draw_kaze(field_points, ton_player, img=def_img, reduction=reduction, is_sanma=is_sanma)

    # 投影変換の計算
    sM = show_img(def_img, m, field_points, dst=dst, reduction=reduction)
    img = def_img.copy()
    isFirst = True
    lose_player = -1

    while (cap.isOpened()):
        # 牌の表示を消す
        show_img(def_img, m, field_points, M=sM, reduction=reduction)
        # 取得映像から牌表示がなくなるまで待機
        for i in range(500):
            if isFirst:
                isFirst = False
                break
            c2 = cv2.waitKey(1)
            if c2 == ord('q'):
                return -1, -1, -1
            elif c2 == ord('p'):
                return -2, -2, -2
        # 再び検出牌を表示
        show_img(img, m, field_points, M=sM, reduction=reduction)
        cv2.waitKey(1)

        # カメラ映像の取得
        ret, im = cap.read()
        im = trans.transform_camera(im, M=cM)
        new_im = im.copy()
        new_im = draw.draw_player_rect2(field_points, win_player, size, new_im)
        cv2.imshow("Camera", cv2.resize(new_im, (1920, 1080)))

        # 上がり牌の検出
        hai_images = []
        for i in range(4-is_sanma):
            hai_images.append(get.get_wintile(field_points, i, im, size))
        win_evals = eval.multi_win_eval(hai_images, 0.8)
        for i, win_eval in enumerate(win_evals):
            [win_class, win_score, win_box] = win_eval
            if len(win_class) > 0 and win_class != 37:
                lose_player = i
                print('set ok')
                music.play_music(LOAD_SE)
                break
        # 上がり牌がある場合，検出を終了
        if lose_player >= 0:
            break
        # 検出結果の表示
        if is_eval_draw:
            hai_img = get.get_hand(field_points, win_player, im, size)
            hand_classes, hand_scores, hand_boxes = eval.hand_eval(hai_img, 0.3)
            hai_img = get.get_dora(field_points, win_player, im, size)
            dora_classes, dora_scores, dora_boxes = eval.dora_eval(hai_img, 0.5)
            hai_img = get.get_naki(field_points, win_player, im, size)
            naki_classes, naki_scores, naki_boxes = eval.naki_eval(hai_img, 0.5)
            img = draw.draw_hand(field_points, hand_classes, hand_boxes, win_player, size, def_img.copy(), reduction=reduction)
            img = draw.draw_dora(field_points, dora_classes, dora_boxes, win_player, size, img, reduction=reduction)
            img = draw.draw_naki(field_points, naki_classes, naki_boxes, win_player, size, img, reduction=reduction)
        c = cv2.waitKey(1)
        # トリガー検出ミスの場合
        if c == ord('q'):
            return -1, -1, 0
        # 点数計算をスキップしたい場合
        if c == ord('p'):
            return -2, -2, 0
    return win_class, win_box, lose_player

def ryukyoku(cap, field_points, size, cM, ton_player, m, dst, player_points, reduction=1, is_sanma=False):
    music.stop_music()
    img = draw.draw_kaze(field_points, ton_player, reduction=reduction, is_sanma=is_sanma)
    img = draw.draw_player_points(field_points, player_points, img=img, reduction=reduction,is_sanma=is_sanma)
    for i in range(4-is_sanma):
        [pt1, pt2] = draw.draw_player_hand(field_points, i, size, reduction=reduction)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), int(3//reduction))

    # 投影変換の計算
    sM = show_img(img, m, field_points, dst=dst, reduction=reduction)

    # 流局の判定
    is_tenpai = [False, False, False, False]
    is_check = [False, False, False, False]
    t_count = [0, 0, 0, 0]
    skip_count=0

    while (cap.isOpened()):
        # カメラ映像の取得
        ret, im = cap.read()
        im = trans.transform_camera(im, M=cM)
        # カメラ映像の表示
        cv2.imshow("Camera", cv2.resize(im, (1920, 1080)))

        # 流局の判定
        images=[]
        for i in range(4-is_sanma):
            hai_img = get.get_hand(field_points, i, im, size)
            images.append(hai_img)
        ryukyoku_eval = eval.mulri_ryukyoku_eval(images)
        for i, ryukyoku in enumerate(ryukyoku_eval):
            if not is_check[i]:
                if ryukyoku == 1:
                    if t_count[i]<0:
                        t_count[i]=0
                    t_count[i] += 1
                    if t_count[i] > 10:
                        is_tenpai[i] = True
                        is_check[i] = True
                        [pt1, pt2] = draw.draw_player_hand(field_points, i, size, reduction=reduction)
                        cv2.rectangle(img, pt1, pt2, (255, 255, 0), int(3//reduction))
                        show_img(img, m, field_points, dst=dst, reduction=reduction, M=sM)
                        music.play_se(TENPAI_SE[1])
                elif ryukyoku==2:
                    if t_count[i]>0:
                        t_count[i]=0
                    t_count[i]-=1
                    if t_count[i] < -10:
                        is_tenpai[i] = False
                        is_check[i] = True
                        [pt1, pt2] = draw.draw_player_hand(field_points, i, size, reduction=reduction)
                        cv2.rectangle(img, pt1, pt2, (0, 0, 255), int(3//reduction))
                        show_img(img, m, field_points, dst=dst, reduction=reduction, M=sM)
                        music.play_se(TENPAI_SE[0])
                else:
                    t_count[i] = 0
        # 間違えていた場合，戻る
        c = cv2.waitKey(1)
        if skip_count<50:
            skip_count+=1
        else:
            if c == ord('q'):
                return player_points, -1
                
        # すべて確認した場合，終了
        if sum(is_check) == 4-is_sanma:
            break
    # ノーテン罰符の支払い
    if sum(is_tenpai) != 4-is_sanma and sum(is_tenpai) != 0:
        not_tenpai_point=(3000-1000*is_sanma)
        for i in range(4-is_sanma):
            if is_tenpai[i]:
                player_points[i] += not_tenpai_point // sum(is_tenpai)
            else:
                player_points[i] -= not_tenpai_point // (4-is_sanma-sum(is_tenpai))
    return player_points, not is_tenpai[ton_player]
            

def draw_movie(field_points, size, m, cap, win_player, cM, agari, dst, min_size=(540, 960, 3)):

    reduction = size[0]/min_size[0]
    min_field = field_points
    img = np.zeros(min_size, np.uint8)
    min_sM = show_img(img, m, field_points, dst=dst, reduction=reduction)
    if agari < 4:
        video = AGARI_VIDEOS[agari]
        video = cv2.VideoCapture(AGARI_VIDEOS[agari])
        count = 0
        while (cap.isOpened()):
            ret, im = cap.read()
            im = trans.transform_camera(im, M=cM)

            cv2.imshow("Camera", cv2.resize(im, (1920, 1080)))
            img = np.zeros(min_size, np.uint8)
            frame = draw.back_place(video, img, min_field, win_player, count, reduction=reduction)
            if frame is None:
                video.release()
                return
            show_img(frame, m, min_field, M=min_sM, reduction=reduction)
            c = cv2.waitKey(1)
            count += 2
            if c == ord('q'):
                video.release()
                return
    # 役満の場合
    else:
        for i in range(len(YAKUMAN_VIDEOS)):
            video = YAKUMAN_VIDEOS[i]
            video = cv2.VideoCapture(video)
            if i == 2:
                music.play_music("./music/和風ジングル.mp3")
            # 動画フレーム
            count = 0
            while (cap.isOpened()):
                ret, im = cap.read()
                im = trans.transform_camera(im, M=cM)
                cv2.imshow("Camera", cv2.resize(im, (1920, 1080)))
                img = np.zeros(min_size, np.uint8)
                frame = draw.back_place(video, img, min_field, win_player, count, reduction=reduction)
                if frame is None:
                    break
                show_img(frame, m, min_field, M=min_sM, reduction=reduction)
                count += 3
                c = cv2.waitKey(1)
                if c == ord('q'):
                    video.release()
                    return
            video.release()

    video.release()
    return


def mahjong_main(cap, m, dst, ton_player, field_points, cM, size, player_points, min_size=(540, 960, 3), round_wind=0, honba=0, kyotaku=0, is_sanma=False, sanma_options=[None,None]):
    # 表示倍率
    reduction = size[0]/min_size[0]

    # 目印用表示
    img = draw.draw_rect(field_points, size, reduction=reduction)
    img = draw.draw_ura_rect(field_points, size, img, reduction=reduction)
    img = draw.draw_kaze(field_points, ton_player, img=img, reduction=reduction, is_sanma=is_sanma)
    img = draw.draw_player_points(field_points, player_points, img=img, reduction=reduction,is_sanma=is_sanma)
    # img=draw.draw_honba(field_points,ton_player,round_wind,honba,img=img,reduction=reduction)

    # 投影
    sM = show_img(img, m, field_points, dst=dst, reduction=reduction)

    # サイコロ用変数
    dice_count = 0
    dice_rand = random.randint(20, 30)
    dice_number = [0, 0]

    # サイコロ
    while (1):
        ret, im = cap.read()
        im = trans.transform_camera(im, M=cM)
        cv2.imshow('Camera', cv2.resize(im, (1920, 1080)))

        # サイコロの表示
        if dice_count < 2*dice_rand and dice_count % 2 == 0:
            dice_img, dice_number = draw.draw_dice(field_points, size, img, reduction, dice_number)
            show_img(dice_img, m, field_points, dst=dst, reduction=reduction, M=sM)
        dice_count += 1
        c = cv2.waitKey(1)
        # 準備完了
        if c == ord('q'):
            break
        # 終了
        elif c == ord('p'):
            return 0, False, kyotaku

    isRead = True
    isRiichi = [False, False, False, False]
    while (isRead):
        reduction = size[0]/min_size[0]
        # トリガー検出
        win_player,isRiichi = read_trigger(cap, field_points, size, cM, ton_player, m, round_wind, honba, kyotaku=kyotaku, dst=dst,
                                  player_points=player_points, reduction=reduction, is_sanma=is_sanma)
        # リーチ分を加算
        kyotaku+=sum(isRiichi)
        # 流局
        if win_player < 0:
            player_points, is_change = ryukyoku(cap, field_points, size, cM, ton_player, m, dst, player_points, reduction=reduction, is_sanma=is_sanma)
            # トリガー検出へ戻る
            if is_change == -1:
                kyotaku-=sum(isRiichi)
                for i in range(4-is_sanma):
                    player_points[i] += 1000*isRiichi[i]
                continue
            # 流局
            return 2*is_change, True, kyotaku

        # 検出牌の表示用フラグ
        draw_flag = False
        read_size = (1080, 1920, 3)
        while (isRead):
            # 検出牌の表示
            if draw_flag:
                img = draw.draw_hand(field_points, hand_classes, hand_boxes, win_player, size, reduction=size[0]/read_size[0])
                img = draw.draw_dora(field_points, dora_classes, dora_boxes, win_player, size, img, reduction=size[0]/read_size[0])
                img = draw.draw_naki(field_points, naki_classes, naki_boxes, win_player, size, img, reduction=size[0]/read_size[0])
                img = draw.draw_wintile(field_points, win_class, win_box, win_player, size, img, reduction=size[0]/read_size[0])
            else:
                img = None
            
            # 待機
            wait_no_wintile(field_points, win_player, size, dst, cap, cM, m, img,
                            reduction=size[0]/read_size[0], is_sanma=is_sanma)

            # 牌配置待機
            win_class, win_box, lose_player = read_wintile(field_points, win_player, size, cap, cM, ton_player,
                                                           m, dst=dst, is_eval_draw=True, reduction=size[0]/read_size[0], is_sanma=is_sanma)
            # トリガー検出へ戻る
            if win_class == -1:
                kyotaku-=sum(isRiichi)
                for i in range(4-is_sanma):
                    player_points[i] += 1000*isRiichi[i]
                break
            # 点数スキップ
            if win_class == -2:
                if ton_player == win_player:
                    return 0, True, 0
                else:
                    return 1, True, 0

            time.sleep(1)
            def_img = draw.draw_player_rect(field_points, win_player, size, reduction=reduction)
            def_img = draw.draw_kaze(field_points, ton_player, img=def_img, reduction=reduction)
            show_img(def_img, m, field_points, M=sM, reduction=reduction)
            # 牌の表示を消す
            for i in range(500):
                cv2.waitKey(1)
            ret, im = cap.read()
            im = trans.transform_camera(im, M=cM)
            # 点数計算
            hai_img = get.get_hand(field_points, win_player, im, size)
            hand_classes, hand_scores, hand_boxes = eval.hand_eval(hai_img, 0.3)
            hai_img = get.get_dora(field_points, win_player, im, size)
            dora_classes, dora_scores, dora_boxes = eval.dora_eval(hai_img, 0.6)
            hai_img = get.get_naki(field_points, win_player, im, size)
            naki_classes, naki_scores, naki_boxes = eval.naki_eval(hai_img, 0.6)

            # 検出結果の表示(ターミナル)
            print('agari', MAHJONG_CLASSES[win_class[0]])
            print('hand', hand_scores)
            for hand_class in hand_classes:
                print(MAHJONG_CLASSES[hand_class])
            print('dora', dora_scores)
            for dora_class in dora_classes:
                print(MAHJONG_CLASSES[dora_class])
            print('naki', naki_scores)
            for naki_class in naki_classes:
                print(MAHJONG_CLASSES[naki_class])
                
            # ツモ判定
            is_tsumo = win_player == lose_player
            # 点数計算
            result = mahjong_calculation.mahjong_auto(hand_classes, naki_classes, naki_boxes, dora_classes, dora_boxes, win_class, win_box, get_wind(
                win_player, ton_player), round_wind=round_wind, honba=honba, is_tsumo=is_tsumo, is_sanma=is_sanma, is_riichi=isRiichi[win_player], sanma_options=sanma_options)

            # 点数計算失敗
            draw_flag = True
            if result == -1:
                music.play_music("./music/ビープ音1.mp3")
                continue
            print(result)
            print(result.yaku)
            if type(result.han) is not int:
                music.play_music("./music/ビープ音1.mp3")
                continue

            agari = get.get_agari(result)
            # 演出表示
            if agari != -1:
                draw_movie(field_points, size, m, cap, win_player, cM, agari, dst=dst)
            # 結果の表示
            music.play_music(POINT_SE[agari+1])
            reduction = size[0]/read_size[0]
            img = draw.draw_result(result, field_points, win_player, size, reduction=reduction)
            img = draw.draw_hand(field_points, hand_classes, hand_boxes, win_player, size, img, reduction=reduction)
            img = draw.draw_dora(field_points, dora_classes, dora_boxes, win_player, size, img, reduction=reduction)
            img = draw.draw_naki(field_points, naki_classes, naki_boxes, win_player, size, img, reduction=reduction)
            img = draw.draw_wintile(field_points, win_class, win_box, lose_player, size, img, reduction=reduction)
            img = draw.draw_kaze(field_points, ton_player, img=img, reduction=reduction, is_sanma=is_sanma)
            img = draw.draw_player_points(field_points, player_points, img=img, reduction=reduction,is_sanma=is_sanma)
            _ = show_img(img, m, field_points, dst=dst, reduction=reduction)
            cv2.waitKey(1)

            yaku_list = result.yaku
            while (1):
                ret, im = cap.read()
                im = trans.transform_camera(im, M=cM)
                cv2.imshow("Camera", cv2.resize(im, (1920, 1080)))
                c = cv2.waitKey(1)
                
                # 役の読み上げ
                if yaku_list != 0:
                    yaku_list=music.start_yaku_voice(yaku_list)
                # 役満等の表示
                elif agari != -1 and not pygame.mixer.get_busy():
                    img = draw.draw_agari(agari, field_points, win_player, size, img, reduction=reduction)
                    music.play_music(AGARI_SE[int(agari//3)])
                    _ = show_img(img, m, field_points, dst=dst, reduction=reduction)
                    agari=-1
                    
                
                # 終了
                if c == ord('q'):
                    isRead = False
                    break
                # 点数間違い
                if c == ord('p'):
                    break

    # 点数変更
    if is_tsumo:
        player_points[win_player] += result.cost['main']+result.cost['additional']*(2-is_sanma)
        # 減点
        for i in range(4-is_sanma):
            if i != win_player:
                if i == ton_player:
                    player_points[i] -= result.cost['main']
                else:
                    player_points[i] -= result.cost['additional']
    else:
        player_points[win_player] += result.cost['main']
        player_points[lose_player] -= result.cost['main']
    # 1000点棒の加算
    player_points[win_player] += kyotaku*1000
    print(player_points)
    if ton_player == win_player:
        return 0, True, 0
    else:
        return 1, True, 0


def main():

    # カメラの設定
    m = get_monitors()[1]
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)

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
    _ = show_img(img, m, min_field, dst=dst)
    cv2.waitKey(500)
    a = 50
    def_points = [[field_points[0][0]+a, field_points[0][1]+a], [field_points[1][0]-a, field_points[1][1]-a]]

    # 投影調節
    sum = 30
    isBreak = False
    while (cap.isOpened()):
        ret, im = cap.read()
        im = trans.transform_camera(im, M=cM)
        im[:field_points[0][1], :] = 0  # 上部
        im[field_points[1][1]:, :] = 0  # 下部
        im[:, :field_points[0][0]] = 0  # 左側
        im[:, field_points[1][0]:] = 0  # 右側
        mask = area.get_rect(im, save_im)
        cv2.rectangle(im, def_points[0], def_points[1], (0, 255, 0), 3)
        if len(mask) > 0:
            cv2.polylines(im, [mask], True, (0, 0, 255), 3)
            dst, isBreak = area.get_dst(field_points, mask, dst, max=size[0])
            cv2.imshow("mask",cv2.resize(im,[1920,1080]))
        img = np.zeros(size, np.uint8)
        cv2.rectangle(img, min_field[0], min_field[1], (255, 0, 0), -1)

        _ = show_img(img, m, min_field, dst=dst)

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
    
    # ゲーム開始前の設定
    is_sanma, is_tonpu, end_point, sanma_options = setting_window()
    is_over = False
    if is_sanma:
        player_points = [35000, 35000, 35000]

    # ゲーム開始
    while (isContinue):
        while (ton_player < 4 - is_sanma and isContinue):
            # 局の開始
            win_result, isContinue, kyotaku = mahjong_main(cap, m, dst, ton_player, field_points, cM, size, player_points, round_wind=round_wind, honba=honba, kyotaku=kyotaku, is_sanma=is_sanma, sanma_options=sanma_options)
            # 親の変更
            if win_result > 0:
                ton_player += 1
            honba += 1
            # 本場のリセット
            if win_result == 1:
                honba = 0
            # 飛んだ場合，終了
            if min(player_points) < 0:
                isContinue = False
                break
            if is_over and max(player_points) >= end_point:
                isContinue = False
                break
        if (1-is_tonpu)==round_wind:
            is_over = True
            if max(player_points) >= end_point:
                isContinue = False

        ton_player = 0
        round_wind += 1

        if round_wind > 3:
            round_wind = 0
        print(round_wind)

    cap.release()
    music.play_music("./music/成功音.mp3")
    img = draw.draw_kaze(field_points, ton_player, reduction=reduction, is_sanma=is_sanma)
    img = draw.draw_player_points(field_points, player_points, img=img, reduction=reduction, is_sanma=is_sanma)
    show_img(img, m, field_points, dst=dst, reduction=reduction)
    c = cv2.waitKey()


if __name__ == '__main__':
    main()
