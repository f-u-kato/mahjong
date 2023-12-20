import pygame

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
