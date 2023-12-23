#　点数計算プログラム

import numpy as np
#計算
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.scores import ScoresCalculator
#麻雀牌
from mahjong.tile import TilesConverter
#役, オプションルール
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
#鳴き
from mahjong.meld import Meld
#風(場&自)
from mahjong.constants import EAST, SOUTH, WEST, NORTH
#シャンテン数
from mahjong.shanten import Shanten

calculator = HandCalculator()
WIND_CLASSES=(EAST,SOUTH,WEST,NORTH)

MAHJONG_CLASSES = ("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                   "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                   "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
                   "ton", "nan", "sha", "pe",
                   "haku", "hatsu", "chun",
                   "aka_5m", "aka_5p", "aka_5s",
                   "ura")
# MAHJONG_LABEL_MAP = {1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,  9:  9,
#                     10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,
#                     19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27,
#                     28: 28, 29: 29, 30: 30, 31: 31, 32: 32, 33: 33, 34: 34, 38: 35, 39: 36,
#                     40: 37, 41: 38}
MAHJONG_CLASSES_NUMBER = {"1m":"1", "2m":"2", "3m":"3", "4m":"4", "5m":"5", "6m":"6", "7m":"7", "8m":"8", "9m":"9",
                          "1p":"1", "2p":"2", "3p":"3", "4p":"4", "5p":"5", "6p":"6", "7p":"7", "8p":"8", "9p":"9",
                          "1s":"1", "2s":"2", "3s":"3", "4s":"4", "5s":"5", "6s":"6", "7s":"7", "8s":"8", "9s":"9",
                          "ton":"1", "nan":"2", "sha":"3", "pe":"4",
                          "haku":"5", "hatsu":"6", "chun":"7",
                          "aka_5m":"0", "aka_5p":"0", "aka_5s":"0",
                          "ura":"-1"}

#結果出力用
def print_hand_result(hand_result, agari):
    result = [
        f"{hand_result.han} han, {hand_result.fu} fu",
        f"{hand_result.cost['main']}, {hand_result.cost['additional']}",
        f"{hand_result.yaku}",
        f"agarihai: {agari}"
    ]

    return result


# 向聴数計算用に赤牌を普通の5として扱う
#ここで赤ドラチェックも行う
def shanten_you_list(tehai):
    c = ''
    has_aka_dora = False
    b = ([i for i in tehai if i != '0'])

    if len(tehai) > len(b):
        has_aka_dora = True 
    for i in b:
        c += i

    while len(tehai) > len(c):
        c += '5'
    
    return c, has_aka_dora

#クラス判定
def naki_class(num):
    if len(num)==4:
        return Meld.KAN
    else:
        if num[0]==num[1]:
            return Meld.PON
        else:
            return Meld.CHI

def remove_char(string, char):
    # 文字列中で最初に見つかった一致する文字を一つだけ取り除く
    new_string = string.replace(char, '', 1)
    return new_string
def replace_with_most_common_char(string):
    # 最も多く出現する文字で全ての文字列を置き換える
    new_string = string.replace(string, '*')
    return new_string
def most_common_char(string):
    """
    文字列中で最も多く出現する文字を返す。
    """
    # 辞書を使用して文字の出現回数を数える
    char_count = {}
    for char in string:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    # 出現回数が最大の文字を見つける
    max_count = 0
    max_char = None
    for char, count in char_count.items():
        if count > max_count:
            max_count = count
            max_char = char

    return max_char

def most_common_char(string):
    # 辞書を使用して文字の出現回数を数える
    char_count = {}
    for char in string:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    # 出現回数が最大の文字を見つける
    max_count = 0
    max_char = None
    for char, count in char_count.items():
        if count > max_count:
            max_count = count
            max_char = char
    return max_char,max_count

#鳴き牌の調節
def naki_correction(hai,locKan):
    if locKan:
        while(len(hai)<4):
            hai+=hai[0]
    naki=naki_class(hai)
    if len(hai)==2:
        if naki=='chi':
            if abs(int(hai[0])-int(hai[1]))>1:
                hai+=str(int(hai[1])+int((int(hai[0])-int(hai[1]))//2))
        elif naki=='pon':
            hai+=hai[0]

    return naki,hai
        

def mahjong_naki(classes, boxes):
    '''
    classes: [0, 1, 2, 3, 4, 5, 6, ...]の長さ14のリスト
    boxes: [[x1, y1, x2, y2], [...], ...]の長さ14のリスト
    x1, y1: バウンディングボックスの左上の座標値
    x2, y2: バウンディングボックスの右下の座標値
    '''
    if len(classes)==0:
        return None,False,["","","",""]
    meld=[]
    man,pin,sou,honor = "","","",""

    classes, boxes = zip(*sorted(zip(classes, boxes), key=lambda x: x[1][1]))

    # mahjongライブラリの入力の仕様にあわせる
    # ライブラリの仕様　man=123 のように種類ごとの数字を記載
    # Meld(Meld.KAN, TilesConverter.string_to_136_array(man='2222'), False),
    box_point=None
    isKan=True
    has_aka_dora=False
    locKan=False
    all_man=''
    all_pin=''
    all_sou=''
    all_honor=''

    for c,box in zip(classes,boxes):
        if box_point is None:
            box_point=box[3]
        else:
            if box_point < box[1]:
                if len(man)>1:
                    naki,man=naki_correction(man,locKan)
                    meld.append(Meld(naki, TilesConverter.string_to_136_array(man=man.replace('0','5')), isKan))
                    if naki=='kan':
                        man=man[:3]
                    all_man+=man
                elif len(pin)>1:
                    naki,pin=naki_correction(pin,locKan)
                    meld.append(Meld(naki_class(pin), TilesConverter.string_to_136_array(pin=pin.replace('0','5')), isKan))
                    if naki=='kan':
                        pin=pin[:3]
                    all_pin+=pin
                elif len(sou)>1:
                    naki,sou=naki_correction(sou,locKan)
                    meld.append(Meld(naki_class(sou), TilesConverter.string_to_136_array(sou=sou.replace('0','5')), isKan))
                    if naki=='kan':
                        sou=sou[:3]
                    all_sou+=sou
                elif len(honor)>1:
                    naki,honor=naki_correction(honor,locKan)
                    meld.append(Meld(naki_class(honor), TilesConverter.string_to_136_array(honors=honor), isKan))
                    if naki=='kan':
                        honor=honor[:3]
                    all_honor+=honor
                man,pin,sou,honor = "","","",""
                box_point=box[3]
                isKan=True
                locKan=False
            elif box_point<box[3]:
                box_point=box[3]

        mc = MAHJONG_CLASSES[c]
        
        if mc[-1] == 'm':
            man += MAHJONG_CLASSES_NUMBER[mc]
        elif mc[-1] == 'p':
            print(mc)
            pin += MAHJONG_CLASSES_NUMBER[mc] 
        elif mc[-1] == 's':
            sou += MAHJONG_CLASSES_NUMBER[mc]        
        else:
            if mc != "ura":
                honor += MAHJONG_CLASSES_NUMBER[mc]
            else:
                isKan=False
        if MAHJONG_CLASSES_NUMBER[mc]==0:
            has_aka_dora=True

    if len(man)>1:
        naki,man=naki_correction(man,locKan)
        meld.append(Meld(naki, TilesConverter.string_to_136_array(man=man), isKan))
        if naki=='kan':
            man=man[:3]
        all_man+=man
    elif len(pin)>1:
        naki,pin=naki_correction(pin,locKan)
        meld.append(Meld(naki_class(pin), TilesConverter.string_to_136_array(pin=pin), isKan))
        if naki=='kan':
            pin=pin[:3]
        all_pin+=pin
    elif len(sou)>1:
        naki,sou=naki_correction(sou,locKan)
        meld.append(Meld(naki_class(sou), TilesConverter.string_to_136_array(sou=sou), isKan))
        if naki=='kan':
            sou=sou[:3]
        all_sou+=sou
    elif len(honor)>1:
        naki,honor=naki_correction(honor,locKan)
        meld.append(Meld(naki_class(honor), TilesConverter.string_to_136_array(honors=honor), isKan))
        if naki=='kan':
            honor=honor[:3]
        all_honor+=honor

    return meld,has_aka_dora,[all_man,all_pin,all_sou,all_honor]

        
        


# ドラ牌認識
def mahjong_dora(classes, boxes):
    dora_indicators = []
    is_riichi=False
    is_ippatu=True
    nuki_dora=0
    #ドラ牌検牌（手配のではなく）
    for class_num,box in zip(classes,boxes):
        mc = MAHJONG_CLASSES[class_num]
        if mc[-1] == 'm':
            if mc[0] =='a':
                dora_indicators.append(TilesConverter.string_to_136_array(man='5')[0])
            else:
                dora_indicators.append(TilesConverter.string_to_136_array(man=MAHJONG_CLASSES_NUMBER[mc])[0])         
        elif mc[-1] == 'p':
            if mc[0] == 'a':
                dora_indicators.append(TilesConverter.string_to_136_array(pin='5')[0])
            else:
                dora_indicators.append(TilesConverter.string_to_136_array(pin=MAHJONG_CLASSES_NUMBER[mc])[0])
        elif mc[-1] == 's':
            if mc[0] =='a':
                dora_indicators.append(TilesConverter.string_to_136_array(sou='5')[0])
            else:
                dora_indicators.append(TilesConverter.string_to_136_array(sou=MAHJONG_CLASSES_NUMBER[mc])[0])    
        else:
            if mc != "ura": 
                dora_indicators.append(TilesConverter.string_to_136_array(honors=MAHJONG_CLASSES_NUMBER[mc])[0])
            else:
                nuki_dora+=1
        if box[3]-box[1]<box[2]-box[0]:
            is_riichi=True
        if box[3]-box[1]>box[2]-box[0]:
            is_ippatu=False
    is_ippatu=is_riichi and is_ippatu
    return dora_indicators,is_riichi,is_ippatu,nuki_dora

def mahjong_hand(classes,win,melds_tiles):
    class_list = []
    man,pin,sou,honor = "","","",""

    classes=np.append(win,classes)
    melds_len=0
    for tiles in melds_tiles:
        melds_len+=len(tiles)


    # mahjongライブラリの入力の仕様にあわせる
    for i,c in enumerate(classes):
        if i>13-melds_len:
            print(i,'break')
            break
        class_list.append(MAHJONG_CLASSES[c])
        mc = MAHJONG_CLASSES[c]
        if mc[-1] == 'm':
            man += MAHJONG_CLASSES_NUMBER[mc]
        elif mc[-1] == 'p':
            pin += MAHJONG_CLASSES_NUMBER[mc] 
        elif mc[-1] == 's':
            sou += MAHJONG_CLASSES_NUMBER[mc]        
        else:
            if mc != "ura":
                honor += MAHJONG_CLASSES_NUMBER[mc]
        
    man+=melds_tiles[0]
    pin+=melds_tiles[1]
    sou+=melds_tiles[2]
    honor+=melds_tiles[3]

    # 手牌14枚
    s_man, aka_man = shanten_you_list(man)
    s_pin, aka_pin = shanten_you_list(pin)
    s_sou, aka_sou = shanten_you_list(sou)
    s_honor, aka_honor = shanten_you_list(honor)    
    has_aka_dora = aka_man or aka_pin or aka_sou or aka_honor
    tiles = TilesConverter.string_to_136_array(man=man, pin=pin, sou=sou,honors=honor,has_aka_dora=has_aka_dora)

    return tiles,has_aka_dora

def mahjong_win(win_class,box):
    has_aka_dora=False
    is_rinshan=False
    mc = MAHJONG_CLASSES[win_class]
    if mc[-1] == 'm':
        if mc[0] =='a':
            win_tile=TilesConverter.string_to_136_array(man='5')[0]
            has_aka_dora=True
        else:
            win_tile=TilesConverter.string_to_136_array(man=MAHJONG_CLASSES_NUMBER[mc])[0]      
    elif mc[-1] == 'p':
        if mc[0] == 'a':
            win_tile=TilesConverter.string_to_136_array(pin='5')[0]
            has_aka_dora=True
        else:
            win_tile=TilesConverter.string_to_136_array(pin=MAHJONG_CLASSES_NUMBER[mc])[0]
    elif mc[-1] == 's':
        if mc[0] =='a':
            win_tile=TilesConverter.string_to_136_array(sou='5')[0]
            has_aka_dora=True
        else:
            win_tile=TilesConverter.string_to_136_array(sou=MAHJONG_CLASSES_NUMBER[mc])[0] 
    else:
        if mc != "ura": 
            win_tile=TilesConverter.string_to_136_array(honors=MAHJONG_CLASSES_NUMBER[mc])[0]
    if box[3]-box[1]<box[2]-box[0]:
        is_rinshan=True
    return win_tile,has_aka_dora,is_rinshan


def mahjong_auto(hand_classes,naki_classes,naki_boxes,dora_classes,dora_boxes,win_class,win_box,player_wind,round_wind=0,honba=0,is_tsumo=False,is_sanma=False):
    player_wind=WIND_CLASSES[player_wind]
    round_wind=WIND_CLASSES[round_wind]
    melds,naki_aka,add_tiles=mahjong_naki(naki_classes,naki_boxes)
    dora_indicators,is_riichi,is_ippatsu,nuki_dora=mahjong_dora(dora_classes,dora_boxes)
    win_tile,win_aka,is_rinshan=mahjong_win(win_class[0],win_box[0])
    tiles,hand_aka=mahjong_hand(hand_classes,win_class[0],add_tiles)
    has_aka_dora=hand_aka | naki_aka | win_aka
    config = HandConfig(is_riichi = is_riichi,is_tsumo = is_tsumo,player_wind=player_wind,round_wind=round_wind,is_ippatsu=is_ippatsu,is_rinshan=is_rinshan,options=OptionalRules(has_open_tanyao=True, has_aka_dora=has_aka_dora,fu_for_open_pinfu=False))
    print(melds,win_tile,tiles)
    try:
        print(dora_indicators)
        result = calculator.estimate_hand_value(tiles, win_tile, melds, dora_indicators, config)
        print(result)
        #抜きドラ計算
        if is_sanma:
            print(result.han)
            result.han+=nuki_dora 
            scores_calculator = ScoresCalculator()
            print("after",result.han)
            result.cost = scores_calculator.calculate_scores(result.han, result.fu, config, len(result.yaku) > 0)
            print(result.cost)
            add_cost=result.cost["additional"]//2
            result.cost["main"]+=add_cost
            result.cost["additional"]+=add_cost
        #本場計算
        if is_tsumo:
            result.cost['additional']+=100*honba
            result.cost["main"]+=100*honba
        else:
            result.cost['main']+=300*honba
    except Exception as e:
        print(e)
        return -1
    return result




def test_result():
    #アガリ形(赤ドラは0,またはrを用いる(並び順はなんでもOK), has_aka_dora=Trueに変更)
    tiles = TilesConverter.string_to_136_array(man='022246', pin='333', sou='33567', has_aka_dora=True)

    #アガリ牌(マンズの6)
    win_tile = TilesConverter.string_to_136_array(man='6')[0]

    #鳴き(チー:CHI, ポン:PON, カン:KAN(True:ミンカン,False:アンカン), カカン:CHANKAN, ヌキドラ:NUKI)
    melds = [
        Meld(Meld.KAN, TilesConverter.string_to_136_array(man='2222'), False),
        Meld(Meld.PON, TilesConverter.string_to_136_array(pin='333')),
        Meld(Meld.CHI, TilesConverter.string_to_136_array(sou='567'))
    ]

    #ドラ(なし)
    dora_indicators = None

    #オプション(ツモ, リンシャンカイホウ, 喰いタン・赤ドラルールを追加)
    config = HandConfig(is_tsumo=True,is_rinshan=True, options=OptionalRules(has_open_tanyao=True, has_aka_dora=True))


    #計算
    result = calculator.estimate_hand_value(tiles, win_tile, melds, dora_indicators, config)
    return result

if __name__=='__main__':
    print(test_result())
