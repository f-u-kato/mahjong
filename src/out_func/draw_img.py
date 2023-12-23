import random
import cv2
import numpy as np

import src.get_func.get_img as get

MAHJONG_CLASSES = ("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                   "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                   "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
                   "ton", "nan", "sha", "pe",
                   "haku", "hatsu", "chun",
                   "aka_5m", "aka_5p", "aka_5s",
                   "ura")

NUM_IMAGES = ['./material/points/0.png', './material/points/1.png', './material/points/2.png', './material/points/3.png', './material/points/4.png', './material/points/5.png',
              './material/points/6.png', './material/points/7.png', './material/points/8.png', './material/points/9.png', './material/points/-.png']

HAN_IMAGE = './material/points/han.png'
FU_IMAGE = './material/points/fu.png'
TEN_IMAGE = './material/points/ten.png'

DICE_IMAGES = []
for i in range(6):
    DICE_IMAGES.append(f'./material/dice/{i+1}.png')

AGARI_IMAGES = ['./material/points/mangan.png', './material/points/haneman.png',
                './material/points/baiman.png', './material/points/3bai.png', './material/points/yakuman.png']


MAHJONG_IMAGES = []
for i in range(9):
    MAHJONG_IMAGES.append(f'./material/manzu/p_ms{i+1}_1.png')
for i in range(9):
    MAHJONG_IMAGES.append(f'./material/pinzu/p_ps{i+1}_1.png')
for i in range(9):
    MAHJONG_IMAGES.append(f'./material/sozu/p_ss{i+1}_1.png')
MAHJONG_IMAGES += ["./material/tupai/p_ji_e_1.png", "./material/tupai/p_ji_s_1.png", "./material/tupai/p_ji_w_1.png", "./material/tupai/p_ji_n_1.png", "./material/tupai/p_no_1.png",
                   "./material/tupai/p_ji_h_1.png", "./material/tupai/p_ji_c_1.png", "./material/manzu/p_msaka.png", "./material/pinzu/pinaka.png", "./material/sozu/soaka.png", "./material/tupai/ura.png"]


def min_max_xy(pt1, pt2):
    [x1, y1] = pt1.copy()
    [x2, y2] = pt2.copy()
    if x1 > x2:
        pt1[0] = x2
        pt2[0] = x1
    if y1 > y2:
        pt1[1] = y2
        pt2[1] = y1
    return pt1, pt2


def padding_img_size(img, size=[544, 967]):
    color = (163, 213, 106)
    h, w, _ = img.shape
    h_pad = size[0]-h
    w_pad = size[1]-w
    return cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=color)


def padding_img(img, padding_size=[9, 16]):
    color = (163, 213, 106)
    h, w, _ = img.shape
    if w*padding_size[0]//padding_size[1] < h:
        h_pad = 0
        w_pad = h*padding_size[1]//padding_size[0]-w
    else:
        h_pad = w*padding_size[0]//padding_size[1]-h
        w_pad = 0
    return cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=color)


def resize_img(img, field_points, size=1.0):
    return cv2.resize(img, (int((field_points[1][0]-field_points[0][0])*size), int((field_points[1][1]-field_points[0][1])*size)))


def place_img(main_img, add_img, field_points, place_points=None, auto=False):
    h2, w2, c = add_img.shape
    new_img = main_img.copy()

    if place_points is None:
        place_points = [0, 0]
        auto = True
    y_point = place_points[1]
    x_point = place_points[0]

    if auto:
        y_point += field_points[0][1]
        x_point += field_points[0][0]

    if c == 4:
        mask = add_img[:, :, 3]  # アルファチャンネルだけ抜き出す。
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 3色分に増やす。
        mask = mask / 255  # 0-255だと使い勝手が悪いので、0.0-1.0に変更。

        add_img = add_img[:, :, :3]  # アルファチャンネルは取り出しちゃったのでもういらない。

        new_img = new_img.astype(np.float32)
        new_img[y_point:y_point+h2, x_point:x_point+w2] *= 1 - mask  # 透過率に応じて元の画像を暗くする。
        new_img[y_point:y_point+h2, x_point:x_point+w2] += add_img * mask  # 貼り付ける方の画像に透過率をかけて加算。
        new_img = new_img.astype(np.uint8)
    else:
        add_img = add_img[:, :, :3]
        new_img[y_point:y_point+h2, x_point:x_point+w2] = add_img

    return new_img


def loop_movie(field_points, video, size, ton_player=1, img=None, reduction=1, speed=1):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.uint8)
    for i in range(speed-1):
        _, _ = video.read()
    out_img = back_place(video, img, draw_points, ton_player)
    if out_img is None:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        out_img = back_place(video, img, draw_points, ton_player)
    return out_img


def draw_dice(field_points, size=(2160, 3840, 3), img=None, reduction=1, number=[0, 0]):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.int8)
    middle = [draw_points[0][0]+(draw_points[1][0]-draw_points[0][0])//2, draw_points[0][1]+(draw_points[1][1]-draw_points[0][1])//2]
    out_img = img.copy()
    for i in range(len(number)):
        number[i] = (number[i]+random.randint(1, 5)) % 6
        dice = cv2.imread(DICE_IMAGES[number[i]], cv2.IMREAD_UNCHANGED)
        h, w, _ = dice.shape
        dice = cv2.resize(dice, (int(h//reduction*2), int(w//reduction*2)), interpolation=cv2.INTER_AREA)
        h, w, _ = dice.shape
        out_img = place_img(out_img, dice, field_points, [middle[0]-w+(w+w//4)*i, middle[1]-h//2])

    return out_img, number


def draw_riichi(field_points, size=(2160, 3840, 3), img=None, reduction=1):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.int8)
    color = (255, 255, 255, 255)
    hai_size = max(draw_points[1][0]-draw_points[0][0], draw_points[1][1]-draw_points[0][1])//15
    middle = [draw_points[0][0]+(draw_points[1][0]-draw_points[0][0])//2, draw_points[0][1]+(draw_points[1][1]-draw_points[0][1])//2]
    add = hai_size*2-hai_size//3

    img[middle[1]-add:middle[1]+add, middle[0]-add:middle[0]+add] = (img[middle[1]-add:middle[1]+add, middle[0]-add:middle[0]+add].astype(np.float16) * 0.5).astype(np.uint8)

    cv2.rectangle(img, [middle[0]-add, middle[1]-add], [middle[0]+add, middle[1]+add], color, int(int(3//reduction)))
    next_add = hai_size*2-hai_size//3*2
    cv2.rectangle(img, [middle[0]-next_add, middle[1]-next_add], [middle[0]+next_add, middle[1]+next_add], color, int(int(3//reduction)))
    return img


def draw_riichi2(field_points, img, size=(2160, 3840, 3), color=(0, 255, 0)):
    for i in range(4):
        pt1, pt2 = get.get_riichi(field_points,i, get_points=True)
        cv2.rectangle(img, pt1, pt2, color, 3)
    return img


def back_place(video, main_img, field_points, player, time=None, reduction=1, skelton=False):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    if time is not None:
        video.set(cv2.CAP_PROP_POS_FRAMES, time)
    ret, frame = video.read()
    if not ret:
        return None
    if player == 0:
        rotate = cv2.ROTATE_180
        frame = cv2.rotate(frame, rotate)
    elif player == 1:
        rotate = cv2.ROTATE_90_CLOCKWISE
        frame = cv2.rotate(frame, rotate)
    elif player == 3:
        rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
        frame = cv2.rotate(frame, rotate)
    if skelton:
        mask = np.all(frame[:, :, :] < [50, 50, 50], axis=-1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        frame[mask, 3] = 0
    out_img = place_img(main_img, resize_img(frame, draw_points), draw_points)
    return out_img

# win_tile


def draw_rect(field_points, size=(2160, 3840, 3), img=None, color=(0, 255, 0), reduction=1):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.uint8)

    hai_size = max(draw_points[1][0]-draw_points[0][0], draw_points[1][1]-draw_points[0][1])//15
    hai_point = [hai_size*2, hai_size]

    pt1 = [draw_points[0][0]+hai_point[0], draw_points[0][1]+hai_point[1]]
    pt2 = [x + hai_size for x in pt1]
    cv2.rectangle(img, pt1, pt2, color, int(3//reduction))
    pt1 = [draw_points[1][0]-hai_point[0], draw_points[1][1]-hai_point[1]]
    pt2 = [x - hai_size for x in pt1]
    cv2.rectangle(img, pt1, pt2, color, int(3//reduction))
    pt1 = [draw_points[1][0]-hai_point[1], draw_points[0][1]+hai_point[0]]
    pt2 = [pt1[0]-hai_size, pt1[1]+hai_size]
    cv2.rectangle(img, pt1, pt2, color, int(3//reduction))
    pt1 = [draw_points[0][0]+hai_point[1], draw_points[1][1]-hai_point[0]]
    pt2 = [pt1[0]+hai_size, pt1[1]-hai_size]
    cv2.rectangle(img, pt1, pt2, color, int(3//reduction))
    return img


def draw_rect_movie(field_points, cap, size=(2160, 3840, 3), img=None, color=(0, 255, 0), reduction=1, speed=3, is_sanma=False):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.uint8)

    hai_size = max(draw_points[1][0]-draw_points[0][0], draw_points[1][1]-draw_points[0][1])//15
    hai_point = [hai_size*2, hai_size]

    for i in range(speed):
        ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, frame = cap.read()

    frame = cv2.resize(frame, [hai_size*2, hai_size*2])
    mask = np.all(frame[:, :, :] == [0, 0, 0], axis=-1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame[mask, 3] = 0

    pt1 = [draw_points[0][0]+hai_point[0], draw_points[0][1]+hai_point[1]]
    pt2 = [x + hai_size for x in pt1]
    pt1, pt2 = min_max_xy(pt1, pt2)
    pt1 = [pt1[0]+(pt2[0]-pt1[0])//2-hai_size, pt1[1]+(pt2[1]-pt1[1])//2-hai_size]
    img = place_img(img, frame, field_points, pt1)

    pt1 = [draw_points[1][0]-hai_point[0], draw_points[1][1]-hai_point[1]]
    pt2 = [x - hai_size for x in pt1]
    pt1, pt2 = min_max_xy(pt1, pt2)
    pt1 = [pt1[0]+(pt2[0]-pt1[0])//2-hai_size, pt1[1]+(pt2[1]-pt1[1])//2-hai_size]
    img = place_img(img, frame, field_points, pt1)
    
    pt1 = [draw_points[1][0]-hai_point[1], draw_points[0][1]+hai_point[0]]
    pt2 = [pt1[0]-hai_size, pt1[1]+hai_size]
    pt1, pt2 = min_max_xy(pt1, pt2)
    pt1 = [pt1[0]+(pt2[0]-pt1[0])//2-hai_size, pt1[1]+(pt2[1]-pt1[1])//2-hai_size]
    img = place_img(img, frame, field_points, pt1)

    if is_sanma:
        pt1 = [draw_points[0][0]+hai_point[1], draw_points[1][1]-hai_point[0]]
        pt2 = [pt1[0]+hai_size, pt1[1]-hai_size]
        pt1, pt2 = min_max_xy(pt1, pt2)
        pt1 = [pt1[0]+(pt2[0]-pt1[0])//2-hai_size, pt1[1]+(pt2[1]-pt1[1])//2-hai_size]
        img = place_img(img, frame, field_points, pt1)
    return img


def draw_rect2(field_points, size=(2160, 3840, 3), img=None, color=(0, 255, 0)):
    if img is None:
        img = np.zeros(size, np.uint8)
    for i in range(4):
        pt1, pt2 = get.get_trigger(field_points, i+1, size=size, get_points=True)
        cv2.rectangle(img, pt1, pt2, color, 3)
    return img

# 鳴き牌の位置


def draw_player_naki(field_points, player, size=(2160, 3840, 3), img=None, first=False, color=(0, 255, 0), return_points=False, reduction=1):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.uint8)
    hai_size = max(draw_points[1][0]-draw_points[0][0], draw_points[1][1]-draw_points[0][1])//15
    pt1,pt2=[0,0],[0,0]
    # 鳴き牌
    if player == 0:
        pt1 = [draw_points[0][0]+hai_size, draw_points[0][1]]
        pt2 = [pt1[0]+hai_size*3, pt1[1]+hai_size*3]
    elif player == 3:
        pt1 = [draw_points[1][0], draw_points[0][1]+hai_size]
        pt2 = [pt1[0]-hai_size*3, pt1[1]+hai_size*3]
    elif player == 2:
        pt1 = [draw_points[1][0]-hai_size, draw_points[1][1]]
        pt2 = [pt1[0]-hai_size*3, pt1[1]-hai_size*3]
    elif player == 1:
        pt1 = [draw_points[0][0], draw_points[1][1]-hai_size]
        pt2 = [pt1[0]+hai_size*3, pt1[1]-hai_size*3]
    return [pt1, pt2]

# 手牌の位置


def draw_player_hand(field_points, player, size=(2160, 3840, 3), img=None, first=False, color=(0, 255, 0), return_points=False, reduction=1):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.uint8)
    hai_size = max(draw_points[1][0]-draw_points[0][0], draw_points[1][1]-draw_points[0][1])//15
    # 鳴き牌
    [pt1, pt2] = draw_player_naki(field_points, player, size, img, first, color, return_points, reduction)
    # 手牌
    if player == 0:
        pt1 = [pt2[0]+hai_size//3, pt1[1]]
        pt2 = [pt1[0]+hai_size*8, pt1[1]+hai_size*4//3]
    elif player == 3:
        pt1 = [pt1[0], pt2[1]+hai_size//3]
        pt2 = [pt1[0]-hai_size*4//3, pt1[1]+hai_size*8]
    elif player == 2:
        pt1 = [pt2[0]-hai_size//3, pt1[1]]
        pt2 = [pt1[0]-hai_size*8, pt1[1]-hai_size*4//3]
    elif player == 1:
        pt1 = [pt1[0], pt2[1]-hai_size//3]
        pt2 = [pt1[0]+hai_size*4//3, pt1[1]-hai_size*8]
    return [pt1, pt2]


def draw_player_dora(field_points, player, size=(2160, 3840, 3), img=None, first=False, color=(0, 255, 0), return_points=False, reduction=1):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.uint8)
    hai_size = max(draw_points[1][0]-draw_points[0][0], draw_points[1][1]-draw_points[0][1])//15
   # 手牌
    [pt1, pt2] = draw_player_hand(field_points, player, size, img, first, color, return_points, reduction)
    # ドラ
    if player == 0:
        pt1 = [pt2[0], pt2[1]+hai_size//10]
        pt2 = [pt1[0]-hai_size*5, pt1[1]+hai_size*4//3]
    elif player == 3:
        pt1 = [pt2[0]-hai_size//10, pt2[1]]
        pt2 = [pt1[0]-hai_size*4//3, pt1[1]-hai_size*5]
    elif player == 2:
        pt1 = [pt2[0], pt2[1]-hai_size//10]
        pt2 = [pt1[0]+hai_size*5, pt1[1]-hai_size*4//3]
    elif player == 1:
        pt1 = [pt2[0]+hai_size//10, pt2[1]]
        pt2 = [pt1[0]+hai_size*4//3, pt1[1]+hai_size*5]
    return [pt1, pt2]


def draw_player_wintile(field_points, player, size=(2160, 3840, 3), img=None, first=False, color=(0, 255, 0), return_points=False, reduction=1):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.uint8)
    hai_size = max(draw_points[1][0]-draw_points[0][0], draw_points[1][1]-draw_points[0][1])//15
   # ドラ
    [pt1, pt2] = draw_player_dora(field_points, player, size, img, first, color, return_points, reduction)
    # あがりはい
    if player == 0:
        pt1 = [pt2[0]-hai_size, pt1[1]]
        pt2 = [pt1[0]-hai_size*4//3, pt1[1]+hai_size*4//3]
    elif player == 3:
        pt1 = [pt1[0], pt2[1]-hai_size]
        pt2 = [pt1[0]-hai_size*4//3, pt1[1]-hai_size*4//3]
    elif player == 2:
        pt1 = [pt2[0]+hai_size, pt1[1]]
        pt2 = [pt1[0]+hai_size*4//3, pt1[1]-hai_size*4//3]
    elif player == 1:
        pt1 = [pt1[0], pt2[1]+hai_size]
        pt2 = [pt1[0]+hai_size*4//3, pt1[1]+hai_size*4//3]
    return [pt1, pt2]


def draw_player_rect(field_points, player, size=(2160, 3840, 3), img=None, first=False, color=(0, 255, 0), return_points=False, reduction=1, is_sanma=False):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.uint8)

    first_img = img.copy()
    hai_size = max(draw_points[1][0]-draw_points[0][0], draw_points[1][1]-draw_points[0][1])//15
    points = []
    # 鳴き牌
    [pt1, pt2] = draw_player_naki(field_points, player, size, img, first, color, return_points, reduction)
    cv2.rectangle(img, pt1, pt2, color, int(3//reduction))
    points.append([pt1, pt2])
    # 手牌
    [pt1, pt2] = draw_player_hand(field_points, player, size, img, first, color, return_points, reduction)
    cv2.rectangle(img, pt1, pt2, color, int(3//reduction))
    points.append([pt1, pt2])
    # ドラ
    [pt1, pt2] = draw_player_dora(field_points, player, size, img, first, color, return_points, reduction)
    cv2.rectangle(img, pt1, pt2, color, int(3//reduction))
    points.append([pt1, pt2])
    if first:
        img = first_img
    # あがりはい
    for i in range(4-is_sanma):
        [pt1, pt2] = draw_player_wintile(field_points, i, size, img, first, color, return_points, reduction)
        cv2.rectangle(img, pt1, pt2, color, int(3//reduction))
        if i == player:
            points.append([pt1, pt2])
    if return_points:
        return points
    return img


def draw_player_rect2(field_points, player, size=(2160, 3840, 3), img=None, color=(0, 255, 0)):
    if img is None:
        img = np.zeros(size, np.uint8)
    hai_size = max(field_points[1][0]-field_points[0][0], field_points[1][1]-field_points[0][1])//15
    # 鳴き牌
    pt1, pt2 = get.get_naki(field_points, player, return_point=True)
    cv2.rectangle(img, pt1, pt2, color, 3)

    # 手牌
    pt1, pt2 = get.get_hand(field_points, player, return_point=True)
    cv2.rectangle(img, pt1, pt2, color, 3)

    # ドラ
    pt1, pt2 = get.get_dora(field_points, player, return_point=True)
    cv2.rectangle(img, pt1, pt2, color, 3)

    # あがりはい
    pt1, pt2 = get.get_wintile(field_points, player, return_point=True)
    cv2.rectangle(img, pt1, pt2, color, 3)

    return img


def draw_hand(field_points, classes, boxes, player, size=(2160, 3840, 3), img=None, reduction=1):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.uint8)

    if player == 0:
        rotate = cv2.ROTATE_180
    elif player == 1:
        rotate = cv2.ROTATE_90_CLOCKWISE
    elif player == 3:
        rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        rotate = None
    for class_num, box in zip(np.flipud(classes), np.flipud(boxes)):
        hai_img = cv2.imread(MAHJONG_IMAGES[class_num], -1)
        hai_img = cv2.resize(hai_img, [int(hai_img.shape[1]*1.2//reduction), int(hai_img.shape[0]*1.2//reduction)])
        box = [int(element // reduction) for element in box]
        if rotate is not None:
            hai_img = cv2.rotate(hai_img, rotate)
        img = place_img(img, hai_img, draw_points, points_hand(draw_points, player, box))
    return img


def draw_naki(field_points, classes, boxes, player, size=(2160, 3840, 3), img=None, reduction=1):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.uint8)

    if player == 0:
        rotate = cv2.ROTATE_180
    elif player == 1:
        rotate = cv2.ROTATE_90_CLOCKWISE
    elif player == 3:
        rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        rotate = None
    for class_num, box in zip(np.flipud(classes), np.flipud(boxes)):
        hai_img = cv2.imread(MAHJONG_IMAGES[class_num], -1)
        hai_img = cv2.resize(hai_img, [int(hai_img.shape[1]*1.2//reduction), int(hai_img.shape[0]*1.2//reduction)])
        box = [int(element // reduction) for element in box]
        if rotate is not None:
            hai_img = cv2.rotate(hai_img, rotate)
        img = place_img(img, hai_img, draw_points, points_naki(draw_points, player, box))

    return img


def draw_wintile(field_points, classes, boxes, player, size=(2160, 3840, 3), img=None, reduction=1):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.uint8)

    if player == 0:
        rotate = cv2.ROTATE_180
    elif player == 1:
        rotate = cv2.ROTATE_90_CLOCKWISE
    elif player == 3:
        rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        rotate = None
    for class_num, box in zip(np.flipud(classes), np.flipud(boxes)):
        hai_img = cv2.imread(MAHJONG_IMAGES[class_num], -1)
        hai_img = cv2.resize(hai_img, [int(hai_img.shape[1]*1.2//reduction), int(hai_img.shape[0]*1.2//reduction)])
        box = [int(element // reduction) for element in box]
        if rotate is not None:
            hai_img = cv2.rotate(hai_img, rotate)
        img = place_img(img, hai_img, draw_points, points_wintile(draw_points, player, box))

    return img


def draw_dora(field_points, classes, boxes, player, size=(2160, 3840, 3), img=None, reduction=1):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.uint8)

    if player == 0:
        rotate = cv2.ROTATE_180
    elif player == 1:
        rotate = cv2.ROTATE_90_CLOCKWISE
    elif player == 3:
        rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        rotate = None

    for class_num, box in zip(np.flipud(classes), np.flipud(boxes)):
        hai_img = cv2.imread(MAHJONG_IMAGES[class_num], -1)
        hai_img = cv2.resize(hai_img, [int(hai_img.shape[1]*1.2//reduction), int(hai_img.shape[0]*1.2//reduction)])
        box = [int(element // reduction) for element in box]
        if rotate is not None:
            hai_img = cv2.rotate(hai_img, rotate)
        img = place_img(img, hai_img, draw_points, points_dora(draw_points, player, box))

    return img


def draw_ura_rect(field_points, size=(2160, 3840, 3), img=None, reduction=1):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.int8)
    color = (0, 255, 0, 255)
    points = draw_player_rect(draw_points, 0, return_points=True)
    pt1, pt2 = min_max_xy(points[0][0], points[0][1])
    y1 = pt2[1]
    points = draw_player_rect(draw_points, 1, return_points=True)
    pt1, pt2 = min_max_xy(points[2][0], points[2][1])
    x1 = pt2[0]
    points = draw_player_rect(draw_points, 2, return_points=True)
    pt1, pt2 = min_max_xy(points[0][0], points[0][1])
    y2 = pt1[1]
    points = draw_player_rect(draw_points, 3, return_points=True)
    pt1, pt2 = min_max_xy(points[2][0], points[2][1])
    x2 = pt1[0]
    cv2.rectangle(img, [x1, y1], [x2, y2], color, int(int(3//reduction)))
    return img


# 点数表示
def draw_player_points(field_points, player_points, size=(2160, 3840, 3), img=None, reduction=1, is_sanma=False):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    hai_size = max(draw_points[1][0]-draw_points[0][0], draw_points[1][1]-draw_points[0][1])//15
    if img is None:
        img = np.zeros(draw_size, np.uint8)
    for i in range(4-is_sanma):
        add_img = num_img(player_points[i])
        add_img = cv2.resize(add_img, [int(add_img.shape[1]*0.8//reduction), int(add_img.shape[0]*0.8//reduction)])
        if i == 0:
            add_img = cv2.rotate(add_img, cv2.ROTATE_180)
            pt1 = [draw_points[0][0]+hai_size, draw_points[0][1]]
        elif i == 1:
            add_img = cv2.rotate(add_img, cv2.ROTATE_90_CLOCKWISE)
            pt1 = [draw_points[0][0], draw_points[1][1]-hai_size-add_img.shape[0]]
        elif i == 2:
            pt1 = [draw_points[1][0]-hai_size-add_img.shape[1], draw_points[1][1]-add_img.shape[0]]
        elif i == 3:
            add_img = cv2.rotate(add_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            pt1 = [draw_points[1][0]-add_img.shape[1], draw_points[0][1]+hai_size]
        img = place_img(img, add_img, draw_points, pt1)
    return img


def draw_kaze(field_points, player, size=(2160, 3840, 3), img=None, reduction=1,is_sanma=False):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.uint8)

    hai_size = max(draw_points[1][0]-draw_points[0][0], draw_points[1][1]-draw_points[0][1])//15
    if is_sanma:
        kaze_img = ['./material/wind/higashi.png', './material/wind/minami.png', './material/wind/nishi.png']
    else:
        kaze_img = ['./material/wind/higashi.png', './material/wind/minami.png', './material/wind/nishi.png', './material/wind/kita.png']
    kaze_img = kaze_img[len(kaze_img)-player:]+kaze_img[:len(kaze_img)-player]

    add_img = cv2.imread(kaze_img[0], -1)
    add_img = cv2.rotate(add_img, cv2.ROTATE_180)
    add_img = cv2.resize(add_img, [hai_size, hai_size])
    r_im = place_img(img, add_img, draw_points, [draw_points[0][0], draw_points[0][1]])

    add_img = cv2.imread(kaze_img[1], -1)
    add_img = cv2.rotate(add_img, cv2.ROTATE_90_CLOCKWISE)
    add_img = cv2.resize(add_img, [hai_size, hai_size])
    r_im = place_img(r_im, add_img, draw_points, [draw_points[0][0], draw_points[1][1]-hai_size])

    add_img = cv2.imread(kaze_img[2], -1)
    add_img = cv2.resize(add_img, [hai_size, hai_size])
    r_im = place_img(r_im, add_img, draw_points, [draw_points[1][0]-hai_size, draw_points[1][1]-hai_size])

    if not is_sanma:
        add_img = cv2.imread(kaze_img[3], -1)
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        add_img = cv2.resize(add_img, [hai_size, hai_size])
        r_im = place_img(r_im, add_img, draw_points, [draw_points[1][0]-hai_size, draw_points[0][1]])

    return r_im


def num_img(num):
    minus = False
    if num < 0:
        minus = True
        num = -num
    num = str(num)
    size = (100, 70*len(num)+minus, 4)
    img = np.zeros(size, np.uint8)
    if minus:
        add_img = cv2.imread(NUM_IMAGES[-1], -1)
        img[:, i*70:(i+1)*70] = add_img
    for i, one_num in enumerate(num):
        add_img = cv2.imread(NUM_IMAGES[int(one_num)], -1)
        img[:, i*70:(i+1)*70] = add_img
    return img


def draw_result(result, field_points, player, size=(2160, 3840, 3), img=None, reduction=1):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.uint8)
    hai_size = max(draw_points[1][0]-draw_points[0][0], draw_points[1][1]-draw_points[0][1])//15
    middle = [draw_points[0][0]+(draw_points[1][0]-draw_points[0][0])//2, draw_points[0][1]+(draw_points[1][1]-draw_points[0][1])//2]
    add = hai_size*2-hai_size//3
    start_points = [[middle[0]-add, middle[1]-add], [middle[0]+add, middle[1]+add]]

    add_img = num_img(result.han)
    add_img = cv2.resize(add_img, [int(add_img.shape[1]//reduction), int(add_img.shape[0]//reduction)])
    # はん
    if player == 0:
        add_img = cv2.rotate(add_img, cv2.ROTATE_180)

        pt1 = start_points[1]
        pt1[1] -= add_img.shape[0]
        pt1[0] -= add_img.shape[1]
    elif player == 3:
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        pt1 = [start_points[0][0], start_points[1][1]]
        pt1[1] -= add_img.shape[0]
    elif player == 2:
        pt1 = start_points[0]
    elif player == 1:
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_CLOCKWISE)

        pt1 = [start_points[1][0], start_points[0][1]]
        pt1[0] -= add_img.shape[1]
    r_im = place_img(img, add_img, draw_points, pt1)

    # はんもじ
    add_size = add_img.shape
    add_img = cv2.imread(HAN_IMAGE, -1)
    add_img = cv2.resize(add_img, [int(add_img.shape[1]//reduction), int(add_img.shape[0]//reduction)])
    if player == 0:
        add_img = cv2.rotate(add_img, cv2.ROTATE_180)
        pt1[0] -= add_size[0]
    elif player == 3:
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pt1[1] -= add_size[1]
    elif player == 2:
        pt1[0] += add_size[1]
    elif player == 1:
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_CLOCKWISE)
        pt1[1] += add_size[0]
    r_im = place_img(r_im, add_img, draw_points, pt1)

    # 符
    add_size = add_img.shape
    add_img = num_img(result.fu)
    print(result.fu)
    add_img = cv2.resize(add_img, [int(add_img.shape[1]//reduction), int(add_img.shape[0]//reduction)])
    if player == 0:
        add_img = cv2.rotate(add_img, cv2.ROTATE_180)
        pt1[0] -= add_size[0]//3*4
    elif player == 3:
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pt1[1] -= add_size[1]//3*4
    elif player == 2:
        pt1[0] += add_size[1]
    elif player == 1:
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_CLOCKWISE)
        pt1[1] += add_size[0]
    r_im = place_img(r_im, add_img, draw_points, pt1)
    # 符文字
    add_size = add_img.shape
    add_img = cv2.imread(FU_IMAGE, -1)
    add_img = cv2.resize(add_img, [int(add_img.shape[1]//reduction), int(add_img.shape[0]//reduction)])
    if player == 0:
        add_img = cv2.rotate(add_img, cv2.ROTATE_180)
        pt1[0] -= add_size[1]//3*2
    elif player == 3:
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pt1[1] -= add_size[0]//3*2
    elif player == 2:
        pt1[0] += add_size[1]
    elif player == 1:
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_CLOCKWISE)
        pt1[1] += add_size[0]
    r_im = place_img(r_im, add_img, draw_points, pt1)

    # 点文字
    add_size = add_img.shape
    add_img = cv2.imread(TEN_IMAGE, -1)
    add_img = cv2.resize(add_img, [int(add_img.shape[1]//reduction), int(add_img.shape[0]//reduction)])
    if player == 0:
        add_img = cv2.rotate(add_img, cv2.ROTATE_180)
        pt1[1] -= add_size[0]
    elif player == 3:
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pt1[0] += add_size[1]
    elif player == 2:
        pt1[1] += add_size[0]
    elif player == 1:
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_CLOCKWISE)
        pt1[0] -= add_size[1]
    r_im = place_img(r_im, add_img, draw_points, pt1)
    add_size = add_img.shape
    pt2 = pt1.copy()
    if player == 0:
        pt2[1] -= add_size[0]
    elif player == 3:
        pt2[0] += add_size[1]
    elif player == 2:
        pt2[1] += add_size[0]
    elif player == 1:
        pt2[0] -= add_size[1]
    r_im = place_img(r_im, add_img, draw_points, pt2)

    # 点
    add_size = add_img.shape
    add_img = num_img(result.cost['main'])
    add_img = cv2.resize(add_img, [int(add_img.shape[1]//reduction), int(add_img.shape[0]//reduction)])

    if player == 0:
        add_img = cv2.rotate(add_img, cv2.ROTATE_180)
        pt1[0] += add_size[1]
    elif player == 3:
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pt1[1] += add_size[0]
    elif player == 2:
        add_size = add_img.shape
        pt1[0] -= add_size[1]
    elif player == 1:
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_CLOCKWISE)
        add_size = add_img.shape
        pt1[1] -= add_size[0]
    r_im = place_img(r_im, add_img, draw_points, pt1)

    add_img = num_img(result.cost['additional'])
    add_img = cv2.resize(add_img, [int(add_img.shape[1]//reduction), int(add_img.shape[0]//reduction)])

    if player == 0:
        add_img = cv2.rotate(add_img, cv2.ROTATE_180)
        pt2[0] += add_size[1]
    elif player == 3:
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pt2[1] += add_size[0]
    elif player == 2:
        add_size = add_img.shape
        pt2[0] -= add_size[1]
    elif player == 1:
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_CLOCKWISE)
        add_size = add_img.shape
        pt2[1] -= add_size[0]
    r_im = place_img(r_im, add_img, draw_points, pt2)

    return r_im


def draw_honba(field_points, ton_player, round_wind, honba=0, size=(2160, 3840, 3), img=None, reduction=1):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(size, np.uint8)
    hai_size = max(draw_points[1][0]-draw_points[0][0], draw_points[1][1]-draw_points[0][1])//15
    kaze_img = ['./material/wind/higashi.png', './material/wind/minami.png', './material/wind/nishi.png', './material/wind/kita.png']
    honba_img = './material/wind/honba.png'
    kyoku_img = './material/wind/kyoku.png'
    field_size = [draw_points[1][1]-draw_points[0][1], draw_points[1][0]-draw_points[0][0]]
    middle = [draw_points[0][0]+(draw_points[1][0]-draw_points[0][0])//2, draw_points[0][1]+(draw_points[1][1]-draw_points[0][1])//2]
    add = hai_size*2-hai_size//3*2
    start_points = [[middle[0]-add, middle[1]-add], [middle[0]+add, middle[1]+add]]

    # 場風
    add_img = cv2.imread(kaze_img[round_wind], -1)
    add_img = cv2.resize(add_img, [int(add_img.shape[1]//reduction), int(add_img.shape[0]//reduction)])
    add_img = cv2.rotate(add_img, cv2.ROTATE_180)
    pt1 = start_points[1]
    pt1[1] -= add_img.shape[0]
    pt1[0] -= add_img.shape[1]
    img = place_img(img, add_img, draw_points, pt1)

    # 局数
    pt1[0] -= add_img.shape[1]
    add_img = num_img(ton_player)
    add_img = cv2.resize(add_img, [int(add_img.shape[1]//reduction), int(add_img.shape[0]//reduction)])
    add_img = cv2.rotate(add_img, cv2.ROTATE_180)
    img = place_img(img, add_img, draw_points, pt1)

    # 局
    pt1[0] -= add_img.shape[1]*2
    add_img = cv2.imread(kyoku_img, -1)
    add_img = cv2.resize(add_img, [int(add_img.shape[1]//reduction), int(add_img.shape[0]//reduction)])
    add_img = cv2.rotate(add_img, cv2.ROTATE_180)
    img = place_img(img, add_img, draw_points, pt1)

    # 本場
    pt1[1] -= add_img.shape[0]
    add_img = cv2.imread(honba_img, -1)
    add_img = cv2.resize(add_img, [int(add_img.shape[1]//reduction), int(add_img.shape[0]//reduction)])
    add_img = cv2.rotate(add_img, cv2.ROTATE_180)
    img = place_img(img, add_img, draw_points, pt1)

    # 本場数
    pt1[0] += add_img.shape[1]
    add_img = num_img(honba)
    add_img = cv2.resize(add_img, [int(add_img.shape[1]//reduction), int(add_img.shape[0]//reduction)])
    add_img = cv2.rotate(add_img, cv2.ROTATE_180)
    img = place_img(img, add_img, draw_points, pt1)

    return img


def draw_agari(agari, field_points, player, size=(2160, 3840, 3), img=None, reduction=1):
    draw_points = ([int(field_points[0][0]//reduction), int(field_points[0][1]//reduction)],
                   [int(field_points[1][0]//reduction), int(field_points[1][1]//reduction)])
    draw_size = [int(size[0]//reduction), int(size[1]//reduction), 3]
    if img is None:
        img = np.zeros(draw_size, np.uint8)

    hai_size = max(draw_points[1][0]-draw_points[0][0], draw_points[1][1]-draw_points[0][1])//15
    middle = [draw_points[0][0]+(draw_points[1][0]-draw_points[0][0])//2, draw_points[0][1]+(draw_points[1][1]-draw_points[0][1])//2]
    add = hai_size*2-hai_size
    start_points = [[middle[0]-add, middle[1]-add], [middle[0]+add, middle[1]+add]]
    r_im = img
    add_img = cv2.imread(AGARI_IMAGES[agari], -1)
    add_img = cv2.resize(add_img, [int(add_img.shape[1]//reduction), int(add_img.shape[0]//reduction)])
    if player == 0:
        add_img = cv2.rotate(add_img, cv2.ROTATE_180)
        pt1 = start_points[0]
        pt1[1] -= add_img.shape[0]
        pt1[0] -= add_img.shape[1]
    elif player == 3:
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pt1 = [start_points[1][0], start_points[0][1]]
        pt1[1] -= add_img.shape[0]
    elif player == 2:
        pt1 = start_points[1]
    elif player == 1:
        add_img = cv2.rotate(add_img, cv2.ROTATE_90_CLOCKWISE)
        pt1 = [start_points[0][0], start_points[1][1]]
        pt1[0] -= add_img.shape[1]

    r_im = place_img(r_im, add_img, draw_points, pt1)

    return r_im


def points_naki(field_points, player, boxes, reduction=1):
    hai_size = max(field_points[1][0]-field_points[0][0], field_points[1][1]-field_points[0][1])//15
    img_h = 65
    points = draw_player_rect(field_points, player, return_points=True)
    pt1, pt2 = min_max_xy(points[0][0], points[0][1])
    if player == 0:
        mid = (boxes[2]-boxes[0])//2+boxes[0]
        points = [pt2[0]-mid-hai_size//2, pt2[1]-boxes[1]]
    elif player == 3:
        mid = (boxes[2]-boxes[0])//2+boxes[0]
        points = [pt1[0]+boxes[1]-img_h+hai_size//2, pt2[1]-mid]
    elif player == 2:
        mid = (boxes[2]-boxes[0])//2+boxes[0]
        points = [pt1[0]+mid, pt1[1]+boxes[1]-img_h//2]
    elif player == 1:
        mid = (boxes[2]-boxes[0])//2+boxes[0]
        points = [pt2[0]-boxes[1], pt1[1]+mid]
    return points


def points_hand(field_points, player, boxes):
    hai_size = max(field_points[1][0]-field_points[0][0], field_points[1][1]-field_points[0][1])//15
    img_h = 65
    points = draw_player_rect(field_points, player, return_points=True)
    pt1, pt2 = min_max_xy(points[1][0], points[1][1])
    if player == 0:
        mid = (boxes[2]-boxes[0])//2+boxes[0]
        points = [pt2[0]-mid-hai_size//2, pt2[1]-boxes[1]]
    elif player == 3:
        mid = (boxes[2]-boxes[0])//2+boxes[0]
        points = [pt1[0]+boxes[1]-img_h+hai_size//2, pt2[1]-mid]
    elif player == 2:
        mid = (boxes[2]-boxes[0])//2+boxes[0]
        points = [pt1[0]+mid, pt1[1]+boxes[1]-img_h//2]
    elif player == 1:
        mid = (boxes[2]-boxes[0])//2+boxes[0]
        points = [pt2[0]-boxes[1], pt1[1]+mid]
    return points


def points_dora(field_points, player, boxes):
    hai_size = max(field_points[1][0]-field_points[0][0], field_points[1][1]-field_points[0][1])//15
    img_h = 65
    points = draw_player_rect(field_points, player, return_points=True)
    pt1, pt2 = min_max_xy(points[2][0], points[2][1])
    if player == 0:
        mid = (boxes[2]-boxes[0])//2+boxes[0]
        points = [pt2[0]-mid-hai_size//2, pt2[1]-boxes[1]]
    elif player == 3:
        mid = (boxes[2]-boxes[0])//2+boxes[0]
        points = [pt1[0]+boxes[1]-img_h+hai_size//2, pt2[1]-mid]
    elif player == 2:
        mid = (boxes[2]-boxes[0])//2+boxes[0]
        points = [pt1[0]+mid, pt1[1]+boxes[1]-img_h//2]
    elif player == 1:
        mid = (boxes[2]-boxes[0])//2+boxes[0]
        points = [pt2[0]-boxes[1], pt1[1]+mid]
    return points


def points_wintile(field_points, player, boxes):
    hai_size = max(field_points[1][0]-field_points[0][0], field_points[1][1]-field_points[0][1])//15
    img_h = 65
    points = draw_player_rect(field_points, player, return_points=True)
    pt1, pt2 = min_max_xy(points[3][0], points[3][1])
    if player == 0:
        mid = (boxes[2]-boxes[0])//2+boxes[0]
        points = [pt2[0]-mid-hai_size//2, pt2[1]-boxes[1]]
    elif player == 3:
        mid = (boxes[2]-boxes[0])//2+boxes[0]
        points = [pt1[0]+boxes[1]-img_h+hai_size//2, pt2[1]-mid]
    elif player == 2:
        mid = (boxes[2]-boxes[0])//2+boxes[0]
        points = [pt1[0]+mid, pt1[1]+boxes[1]-img_h//2]
    elif player == 1:
        mid = (boxes[2]-boxes[0])//2+boxes[0]
        points = [pt2[0]-boxes[1], pt1[1]+mid]
    return points


def main():
    p1, p2 = min_max_xy([50, 100], [100, 50])
    print(p1, p2)


if __name__ == '__main__':
    main()
