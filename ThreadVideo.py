

# 映像再生用のスレッドクラス
import threading

import cv2


class ThreadVideo(threading.Thread):
    def __init__(self,field_points,size,ton_player,reduction,round_wind,honba,dst,m):
        threading.Thread.__init__(self)
        self.stopped = False
        self.trigger=cv2.VideoCapture(TRIGGER_VIDEOS[random.randint(0,len(TRIGGER_VIDEOS)-1)])
        self.speed=1
        self.field_points=field_points
        self.size=size
        self.ton_player=ton_player
        self.reduction=reduction
        self.isRiichi=False
        self.r_video=None
        self.round_wind=round_wind
        self.honba=honba
        self.m=m
        img = draw.draw_rect_movie(field_points,self.trigger,size,img=None,reduction=reduction)
        img = draw.draw_kaze(field_points,ton_player,img=img,reduction=reduction)
        img = draw.draw_honba(field_points,ton_player,round_wind,honba,img=img,reduction=reduction)
        self.img = draw.draw_riichi(field_points,img=img,reduction=reduction)
        self.sM=show_img(img,m,field_points,dst=dst,reduction=reduction)
        
        cv2.waitKey(1)

        #音楽再生
        self.rand=random.randint(0,len(PLAY_BGM)-1)
        music.loop_music(PLAY_BGM[self.rand])
        self.back=cv2.VideoCapture(BACK_MOVIES[self.rand])


    def run(self):
        while not self.stopped:
            #背景動画動画
            img=draw.loop_movie(self.field_points,self.back,self.size,self.ton_player,reduction=self.reduction,speed=self.speed)
            #リーチ演出
            if self.isRiichi:
                _,_=self.r_video.read()
                _,_=self.r_video.read()
                tmp_img = draw.back_place(self.r_video,img,self.field_points,self.ton_player,reduction=self.reduction,skelton=True)
                if tmp_img is None:
                    self.isRiichi=False
                    self.r_video.release()
                    music.loop_music(RIICHI_BGM[self.rand])
                else:
                    img=tmp_img
            img = draw.draw_rect_movie(self.field_points,self.trigger,self.size,img=img,reduction=self.reduction)
            img = draw.draw_kaze(self.field_points,self.ton_player,img=img,reduction=self.reduction)
            img = draw.draw_honba(self.field_points,self.ton_player,self.round_wind,self.honba,img=img,reduction=self.reduction)
            img = draw.draw_riichi(self.field_points,img=img,reduction=self.reduction)
            show_img(img,self.m,self.field_points,M=self.sM,reduction=self.reduction)
            cv2.waitKey(1)
            print("",end=".")


        self.back.release()

    def stop(self):
        self.stopped = True

    def riichi(self):
        self.isRiichi=True
        music.stop_music()
        self.r_video=cv2.VideoCapture(RIICHI_VIDEO)
        music.play_music(RIICHI_SE[0])
        self.speed*=2