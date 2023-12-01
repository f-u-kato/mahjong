import cv2


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


def get_agari(result):
    if result.han >= 3:
        if result.han >= 13:
            return 4
        elif result.han >= 11:
            return 3
        elif result.han >= 8:
            return 2
        elif result.han >= 6:
            return 1
        elif (result.han == 3 and result.fu >= 70) or (result.han == 4 and result.fu >= 40) or (result.han == 5):
            return 0
        else:
            return -1
    else:
        return -1


def get_riichi(field_points, player, img=None, get_points=False):
    hai_size = max(field_points[1][0]-field_points[0][0], field_points[1][1]-field_points[0][1])//15
    middle = [field_points[0][0]+(field_points[1][0]-field_points[0][0])//2, field_points[0][1]+(field_points[1][1]-field_points[0][1])//2]
    add = hai_size*2-hai_size//3
    small_add = hai_size*2-hai_size//3*2
    if player == 0:
        pt1 = [middle[0]+small_add, middle[1]-add]
        pt2 = [middle[0]+add, middle[1]+add]
    elif player == 1:
        pt1 = [middle[0]-add, middle[1]+small_add]
        pt2 = [middle[0]+add, middle[1]+add]
    elif player == 2:
        pt1 = [middle[0]-add, middle[1]-add]
        pt2 = [middle[0]-small_add, middle[1]+add]
    elif player == 3:
        pt1 = [middle[0]-add, middle[1]-add]
        pt2 = [middle[0]+add, middle[1]-small_add]
    pt1,pt2=min_max_xy(pt1,pt2)
    if get_points:
        return pt1, pt2
    im = img[pt1[1]:pt2[1], pt1[0]:pt2[0], :]
    return im


def get_trigger(field_points, player, img=None, size=(2160, 3840, 3), color=(0, 255, 0), get_points=False):
    hai_size = max(field_points[1][0]-field_points[0][0], field_points[1][1]-field_points[0][1])//15
    hai_point = [hai_size*2+50, hai_size]
    if player == 0:
        pt1 = [field_points[0][0]+hai_point[0]-hai_size//4, field_points[0][1]+hai_point[1]+hai_size//8]
        pt2 = [x + hai_size for x in pt1]
    elif player == 2:
        pt1 = [field_points[1][0]-hai_point[0]+hai_size//6, field_points[1][1]-hai_point[1]-hai_size//4]
        pt2 = [x - hai_size for x in pt1]
    elif player == 3:
        pt1 = [field_points[1][0]-hai_size-hai_size//4, field_points[0][1]+hai_point[0]-hai_size//4]
        pt2 = [pt1[0]-hai_size, pt1[1]+hai_size]
    else:
        pt1 = [field_points[0][0]+hai_point[1]+hai_size//6, field_points[1][1]-hai_point[0]+hai_size//8]
        pt2 = [pt1[0]+hai_size, pt1[1]-hai_size]
    pt1, pt2 = min_max_xy(pt1, pt2)
    if get_points:
        return pt1, pt2
    im = img[pt1[1]:pt2[1], pt1[0]:pt2[0], :]

    return im


def get_naki(field_points, player, im=None, size=(2160, 3840, 3), return_point=False):
    hai_size = max(field_points[1][0]-field_points[0][0], field_points[1][1]-field_points[0][1])//15
    if player == 0:
        pt1 = [field_points[0][0]+hai_size, field_points[0][1]+hai_size//2-hai_size//3]
        pt2 = [pt1[0]+hai_size*3, pt1[1]+hai_size*3]
        rotate = cv2.ROTATE_180
    elif player == 3:
        pt1 = [field_points[1][0]-hai_size//6, field_points[0][1]+hai_size+hai_size//6]
        pt2 = [pt1[0]-hai_size*3, pt1[1]+hai_size*3]
        rotate = cv2.ROTATE_90_CLOCKWISE
    elif player == 2:
        pt1 = [field_points[1][0]-hai_size-hai_size//4, field_points[1][1]-hai_size//4]
        pt2 = [pt1[0]-hai_size*3+hai_size//4, pt1[1]-hai_size*3]
        rotate = None
    elif player == 1:
        pt1 = [field_points[0][0]+hai_size//6, field_points[1][1]-hai_size-hai_size//6]
        pt2 = [pt1[0]+hai_size*3, pt1[1]-hai_size*3]
        rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
    pt1, pt2 = min_max_xy(pt1, pt2)
    if return_point:
        return pt1, pt2
    img = im[pt1[1]:pt2[1], pt1[0]:pt2[0], :]
    if rotate is not None:
        img = cv2.rotate(img, rotate)

    return img


def get_hand(field_points, player, im=None, size=(2160, 3840, 3), return_point=False):
    hai_size = max(field_points[1][0]-field_points[0][0], field_points[1][1]-field_points[0][1])//15
    if player == 0:
        pt1 = [field_points[0][0]+hai_size, field_points[0][1]+hai_size//2-hai_size//3]
        pt2 = [pt1[0]+hai_size*3, pt1[1]+hai_size*3]
        pt1 = [pt2[0]+hai_size//3, pt1[1]]
        pt2 = [pt1[0]+hai_size*8-hai_size//3, pt1[1]+hai_size*4//3]
        rotate = cv2.ROTATE_180
    elif player == 3:
        pt1 = [field_points[1][0]-hai_size//6, field_points[0][1]+hai_size+hai_size//6]
        pt2 = [pt1[0]-hai_size*3, pt1[1]+hai_size*3]
        pt1 = [pt1[0], pt2[1]+hai_size//3]
        pt2 = [pt1[0]-hai_size*4//3, pt1[1]+hai_size*8-hai_size//6]
        rotate = cv2.ROTATE_90_CLOCKWISE
    elif player == 2:
        pt1 = [field_points[1][0]-hai_size-hai_size//4, field_points[1][1]-hai_size//4]
        pt2 = [pt1[0]-hai_size*3+hai_size//4, pt1[1]-hai_size*3]
        pt1 = [pt2[0]-hai_size//3, pt1[1]]
        pt2 = [pt1[0]-hai_size*8+hai_size//3, pt1[1]-hai_size*4//3]
        rotate = None
    elif player == 1:
        pt1 = [field_points[0][0]+hai_size//6, field_points[1][1]-hai_size-hai_size//3]
        pt2 = [pt1[0]+hai_size*3, pt1[1]-hai_size*3]
        pt1 = [pt1[0], pt2[1]-hai_size//4]
        pt2 = [pt1[0]+hai_size*5//3-hai_size//4, pt1[1]-hai_size*8+hai_size//2]
        rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
    pt1, pt2 = min_max_xy(pt1, pt2)
    if return_point:
        return pt1, pt2
    img = im[pt1[1]:pt2[1], pt1[0]:pt2[0], :]
    if rotate is not None:
        img = cv2.rotate(img, rotate)
    return img


def get_dora(field_points, player, im=None, size=(2160, 3840, 3), return_point=False):
    hai_size = max(field_points[1][0]-field_points[0][0], field_points[1][1]-field_points[0][1])//15
    if player == 0:
        pt1 = [field_points[0][0]+hai_size, field_points[0][1]+hai_size//2-hai_size//3]
        pt2 = [pt1[0]+hai_size*3, pt1[1]+hai_size*3]
        pt1 = [pt2[0]+hai_size//3, pt1[1]]
        pt2 = [pt1[0]+hai_size*8-hai_size//3, pt1[1]+hai_size*4//3]
        pt1 = [pt2[0]-hai_size//5+hai_size//4, pt2[1]+hai_size//10]
        pt2 = [pt1[0]-hai_size*5+hai_size//4, pt1[1]+hai_size*4//3]
        rotate = cv2.ROTATE_180
    elif player == 3:
        pt1 = [field_points[1][0]-hai_size//6, field_points[0][1]+hai_size+hai_size//6]
        pt2 = [pt1[0]-hai_size*3, pt1[1]+hai_size*3]
        pt1 = [pt1[0], pt2[1]+hai_size//3]
        pt2 = [pt1[0]-hai_size*4//3, pt1[1]+hai_size*8-hai_size//6]
        pt1 = [pt2[0]-hai_size//10, pt2[1]]
        pt2 = [pt1[0]-hai_size*4//3, pt1[1]-hai_size*5]
        rotate = cv2.ROTATE_90_CLOCKWISE
    elif player == 2:
        pt1 = [field_points[1][0]-hai_size-hai_size//4, field_points[1][1]-hai_size//4]
        pt2 = [pt1[0]-hai_size*3+hai_size//4, pt1[1]-hai_size*3]
        pt1 = [pt2[0]-hai_size//3, pt1[1]]
        pt2 = [pt1[0]-hai_size*8+hai_size//3, pt1[1]-hai_size*4//3]
        pt1 = [pt2[0], pt2[1]-hai_size//10]
        pt2 = [pt1[0]+hai_size*5-hai_size//3, pt1[1]-hai_size*4//3]
        rotate = None
    elif player == 1:
        pt1 = [field_points[0][0]+hai_size//6, field_points[1][1]-hai_size-hai_size//3]
        pt2 = [pt1[0]+hai_size*3, pt1[1]-hai_size*3]
        pt1 = [pt1[0], pt2[1]-hai_size//3]
        pt2 = [pt1[0]+hai_size*4//3, pt1[1]-hai_size*8+hai_size//2]
        pt1 = [pt2[0]+hai_size//10, pt2[1]]
        pt2 = [pt1[0]+hai_size*4//3, pt1[1]+hai_size*5]
        rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
    pt1, pt2 = min_max_xy(pt1, pt2)
    if return_point:
        return pt1, pt2
    img = im[pt1[1]:pt2[1], pt1[0]:pt2[0], :]
    if rotate is not None:
        img = cv2.rotate(img, rotate)
    return img


def get_wintile(field_points, player, im=None, size=(2160, 3840, 3), return_point=False):
    hai_size = max(field_points[1][0]-field_points[0][0], field_points[1][1]-field_points[0][1])//15
    if player == 0:
        pt1 = [field_points[0][0]+hai_size, field_points[0][1]+hai_size//2-hai_size//3]
        pt2 = [pt1[0]+hai_size*3, pt1[1]+hai_size*3]
        pt1 = [pt2[0]+hai_size//3, pt1[1]]
        pt2 = [pt1[0]+hai_size*8-hai_size//3, pt1[1]+hai_size*4//3]
        pt1 = [pt2[0]-hai_size//5+hai_size//4, pt2[1]+hai_size//10]
        pt2 = [pt1[0]-hai_size*5+hai_size//4, pt1[1]+hai_size*4//3]
        pt1 = [pt2[0]-hai_size, pt1[1]]
        pt2 = [pt1[0]-hai_size*4//3, pt1[1]+hai_size*4//3]
        rotate = cv2.ROTATE_180
    elif player == 3:
        pt1 = [field_points[1][0]-hai_size//6, field_points[0][1]+hai_size+hai_size//6]
        pt2 = [pt1[0]-hai_size*3, pt1[1]+hai_size*3]
        pt1 = [pt1[0], pt2[1]+hai_size//3]
        pt2 = [pt1[0]-hai_size*4//3, pt1[1]+hai_size*8-hai_size//6]
        pt1 = [pt2[0]-hai_size//10, pt2[1]]
        pt2 = [pt1[0]-hai_size*4//3, pt1[1]-hai_size*5]
        pt1 = [pt1[0], pt2[1]-hai_size]
        pt2 = [pt1[0]-hai_size*4//3, pt1[1]-hai_size*4//3]
        rotate = cv2.ROTATE_90_CLOCKWISE
    elif player == 2:
        pt1 = [field_points[1][0]-hai_size-hai_size//4, field_points[1][1]-hai_size//4]
        pt2 = [pt1[0]-hai_size*3+hai_size//4, pt1[1]-hai_size*3]
        pt1 = [pt2[0]-hai_size//3, pt1[1]]
        pt2 = [pt1[0]-hai_size*8+hai_size//3, pt1[1]-hai_size*4//3]
        pt1 = [pt2[0], pt2[1]-hai_size//10]
        pt2 = [pt1[0]+hai_size*5-hai_size//3, pt1[1]-hai_size*4//3]
        pt1 = [pt2[0]+hai_size, pt1[1]]
        pt2 = [pt1[0]+hai_size*4//3, pt1[1]-hai_size*4//3]
        rotate = None
    elif player == 1:
        pt1 = [field_points[0][0]+hai_size//6, field_points[1][1]-hai_size-hai_size//3]
        pt2 = [pt1[0]+hai_size*3, pt1[1]-hai_size*3]
        pt1 = [pt1[0], pt2[1]-hai_size//3]
        pt2 = [pt1[0]+hai_size*4//3, pt1[1]-hai_size*8+hai_size//2]
        pt1 = [pt2[0]+hai_size//10, pt2[1]]
        pt2 = [pt1[0]+hai_size*4//3, pt1[1]+hai_size*5]
        pt1 = [pt1[0], pt2[1]+hai_size]
        pt2 = [pt1[0]+hai_size*4//3, pt1[1]+hai_size*4//3]
        rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
    pt1, pt2 = min_max_xy(pt1, pt2)
    if return_point:
        return pt1, pt2
    img = im[pt1[1]:pt2[1], pt1[0]:pt2[0], :]
    if rotate is not None:
        img = cv2.rotate(img, rotate)
    return img
