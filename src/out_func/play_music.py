import pygame

YAKU_PATH = "./music/yaku/"
YAKU_FILES = {
    "AkaDora": "アカドラ.wav",
    "Chankan": "チャンカン.wav",
    "Chantai": "チャンタ.wav",
    "Chiitoitsu": "チートイツ.wav",
    "Chinitsu": "チンイツ.wav",
    "Chun": "チュン.wav",
    "DaburuOpenRiichi": "ダブルオープンリーチ.wav",
    "DaburuRiichi": "ダブルリーチ.wav",
    "Dora": "ドラ.wav",
    "Haitei": "ハイテイ.wav",
    "Haku": "ハク.wav",
    "Hatsu": "ハツ.wav",
    "Honitsu": "ホンイツ.wav",
    "Honroto": "ホンロートウ.wav",
    "Houtei": "ホウテイ.wav",
    "Iipeiko": "イーペーコー.wav",
    "Ippatsu": "イッパツ.wav",
    "Ittsu": "イッツ.wav",
    "Junchan": "ジュンチャン.wav",
    "NagashiMangan": "ナガシマンガン.wav",
    "OpenRiichi": "オープンリーチ.wav",
    "Pinfu": "ピンフ.wav",
    "Renhou": "レンホウ.wav",
    "Riichi": "リーチ.wav",
    "Rinshan": "リンシャンカイホウ.wav",
    "Ryanpeikou": "リャンペーコー.wav",
    "Sanankou": "サンアンコー.wav",
    "SanKantsu": "サンカンツ.wav",
    "Sanshoku": "サンショクドウジュン.wav",
    "SanshokuDoukou": "サンショクドウコー.wav",
    "Shosangen": "ショウサンゲン.wav",
    "Tanyao": "タンヤオ.wav",
    "Toitoi": "トイトイ.wav",
    "Tsumo": "ツモ.wav",
    # "YakuhaiEast": "ヤクハイ.wav", # この項目は質問には含まれていない
    # "YakuhaiNorth": "ヤクハイ.wav", # この項目は質問には含まれていない
    # "YakuhaiOfPlace": "ヤクハイ.wav", # この項目は質問には含まれていない
    # "YakuhaiOfRound": "ヤクハイ.wav", # この項目は質問には含まれていない
    # "YakuhaiSouth": "ヤクハイ.wav", # この項目は質問には含まれていない
    # "YakuhaiWest": "ヤクハイ.wav", # この項目は質問には含まれていない
}

def loop_music(path):
    pygame.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play(-1)
    return

def play_music(path):
    pygame.init()

    # 音楽ファイルを読み込む
    pygame.mixer.music.load(path)

    # 音楽を再生する
    pygame.mixer.music.play()

    return

def play_se(path):
    pygame.mixer.Sound(path).play()
    return

def stop_music():
    pygame.mixer.music.stop()
    return

def fade_music(ms=1000):
    pygame.mixer.music.fadeout(ms)
    return

def pause_music():
    pygame.mixer.music.pause()
    return

def unpause_music():
    pygame.mixer.music.unpause()
    return

def start_yaku_voice(yaku_list):
    if pygame.mixer.get_busy():
        return yaku_list
    else:
        yaku = yaku_list.pop()
        play_se(YAKU_PATH + YAKU_FILES[yaku])
        return yaku_list
