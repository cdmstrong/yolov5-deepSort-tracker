import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import time
import numpy as np

import objtracker
from objdetector import Detector
import cv2


VIDEO_PATH = './video/short.mp4'
def isRed(img):
    # cv2.imwrite('video/' + time.strftime('%Y-%m-%d-%H-%M-%S') + '.png', img)
    img = cv2.medianBlur(img, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    min = np.array([0, 43, 46])
    max = np.array([10, 255, 255])
    img = cv2.inRange(img, min, max)
    if np.max(img):
        return True
    else :
        return False
def catch_person(isLight, track_id, img, ori):
    if isLight:
        if track_id not in person_list:
            person_list.append(track_id)
            cv2.imwrite('video/catch/' + time.strftime('%Y-%m-%d-%H-%M-%S') + '.png', img)
            cv2.imwrite('video/catch/' + time.strftime('%Y-%m-%d-%H-%M-%S') + str(track_id) + '.png', ori)
            # 
            print('妖秀啦，敢闯红灯')
if __name__ == '__main__':

    # 根据视频尺寸，填充供撞线计算使用的polygon
    width = 1920
    height = 1080
    mask_image_temp = np.zeros((height, width), dtype=np.uint8)

    # 填充第一个撞线polygon（蓝色）
    # list_pts_blue = [[204, 305], [227, 431], [605, 522], [1101, 464], [1900, 601], [1902, 495], [1125, 379], [604, 437],
                    #  [299, 375], [267, 289]]
    list_pts_blue = [[0, 450], [1890, 450], [1890, 550], [0, 550]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)

    # 填充第二个撞线polygon（黄色）
    mask_image_temp = np.zeros((height, width), dtype=np.uint8)
    # list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
    #                    [594, 637], [118, 483], [109, 303]]
    list_pts_yellow = [[0, 600], [1890, 600], [1890, 750], [0, 750]]
    
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=1)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # 撞线检测用的mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_yellow_value_2

    # 缩小尺寸，1920x1080->960x540
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (width//2, height//2))


    # 黄 色盘
    yellow_color_plate = [0, 255, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = yellow_image
    
    # 缩小尺寸，1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (width//2, height//2))

    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []

    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []

    # 下行数量
    down_count = 0
    # 上行数量
    up_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int((width/2) * 0.01), int((height/2) * 0.05))

    # 实例化yolov5检测器
    detector = Detector()

    # 打开视频
    capture = cv2.VideoCapture(VIDEO_PATH)
    person_list = []

    while True:
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break
        isLight = False
        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (width//2, height//2))
        im_ori = im.copy()
        list_bboxs = []
        # # 检测红绿灯
        # red_img, red_box = detector.detect(im, ['traffic light'])
        # if len(red_box) > 0:
        #     x1, y1, x2, y2, lbl, conf = red_box
        #     red_light = isRed(red_img[y1: y2, x1: x2])
        #     if red_light:
        #         print('检测到红灯')
        
        # 更新跟踪器
        output_image_frame, list_bboxs, light = objtracker.update(detector, im)
        # 输出图片
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)
        if light is not None:
            (x1, y1, x2, y2, label, conf) = light
            red_light = isRed(im_ori[y1: y2, x1: x2])
            if red_light:
                print('检测到红灯')
                isLight = True
        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox
                
                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.6))
                # 撞线的点
                y = y1_offset
                x = x1
                if polygon_mask_blue_and_yellow[y, x] == 1:
                    
                    catch_person(isLight, track_id, im_ori[y1: y2, x1: x2], im_ori)
                    # 如果撞 蓝polygon
                    # if track_id not in list_overlapping_blue_polygon:
                    #     list_overlapping_blue_polygon.append(track_id)
                    # 判断 黄polygon list里是否有此 track_id
                    # 有此track_id，则认为是 UP (上行)方向
                    # if track_id in list_overlapping_yellow_polygon:
                        # 上行+1
                        # up_count += 1
                        # print('up count:', up_count, ', up id:', list_overlapping_yellow_polygon)
                        # # 删除 黄polygon list 中的此id
                        # list_overlapping_yellow_polygon.remove(track_id)
                        # catch_person(isLight, track_id, im_ori[y1: y2, x1: x2])
                # elif polygon_mask_blue_and_yellow[y, x] == 2:
                #     # 如果撞 黄polygon
                #     if track_id not in list_overlapping_yellow_polygon:
                #         list_overlapping_yellow_polygon.append(track_id)
                #     # 判断 蓝polygon list 里是否有此 track_id
                #     # 有此 track_id，则 认为是 DOWN（下行）方向
                #     if track_id in list_overlapping_blue_polygon:
                #         # 下行+1
                #         down_count += 1
                #         print('down count:', down_count, ', down id:', list_overlapping_blue_polygon)
                #         # 删除 蓝polygon list 中的此id
                #         list_overlapping_blue_polygon.remove(track_id)
                #         catch_person(isLight, track_id, im_ori[y1: y2, x1: x2])
            # ----------------------清除无用id----------------------
            # list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
            # for id1 in list_overlapping_all:
            #     is_found = False
            #     for _, _, _, _, _, bbox_id in list_bboxs:
            #         if bbox_id == id1:
            #             is_found = True
            #     if not is_found:
            #         # 如果没找到，删除id
            #         if id1 in list_overlapping_yellow_polygon:
            #             list_overlapping_yellow_polygon.remove(id1)

            #         if id1 in list_overlapping_blue_polygon:
            #             list_overlapping_blue_polygon.remove(id1)
            # list_overlapping_all.clear()
            # 清空list
            # list_bboxs.clear()
        else:
            # 如果图像中没有任何的bbox，则清空list
            list_overlapping_blue_polygon.clear()
            list_overlapping_yellow_polygon.clear()
            
        # 输出计数信息
        text_draw = '闯红灯人数: ' + str(len(person_list))
                    
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=0.75, color=(0, 0, 255), thickness=2)
        cv2.imshow('Counting Demo', output_image_frame)
        cv2.waitKey(1)

    capture.release()
    cv2.destroyAllWindows()
