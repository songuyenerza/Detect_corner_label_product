from tabnanny import check
from tkinter.messagebox import RETRY
from detect_yolov5 import *
from operator import itemgetter, attrgetter
import numpy
import numpy as np
from predict_u2net import *
import torch
import cv2
from PIL import Image, ImageOps
import timeit


def sorted_box(box_list):

    #box_1 = topleft, box3 = topright, box4 = b_right, box2 = b_left
    box_list = sorted(box_list, key = lambda box_list: box_list[0])
    box_12 = box_list[:2]
    box_34 = box_list[2:]

    box_12 = sorted(box_12, key = lambda box_12: box_12[1])
    box1 = box_12[0]
    box2 = box_12[1]

    box_34 = sorted(box_34, key = lambda box_34: box_34[1])
    box3 = box_34[0]
    box4 = box_34[1]

    return [box1, box3, box4, box2]

def check_box(box_list, box_true):
    #clear_box: box= false==> convert to rectangle
    box_list = sorted_box(box_list)

    [box1, box3, box4, box2] = box_list

    #angle_line1(top):
    goc_line1 = (box3[1] - box1[1])/(box3[0] - box1[0])
    #angle_line2(Buton)
    goc_line2 = (box4[1] - box2[1])/(box4[0] - box2[0])
    #line_3(right):
    goc_line3 = (box4[1] - box3[1])/(box4[0] - box3[0])
    #line4(left):
    goc_line4 = (box2[1] - box1[1])/(box2[0] - box1[0])

    if abs(goc_line1 - goc_line2) >0.12 or abs(goc_line3 - goc_line4)> 0.12:
        return box_true
    else:
        return box_list


def get_box_crop(box_list):
    box_x = []
    box_y = []
    for box in box_list:
        box_x.append(box[0])
        box_y.append(box[1])
    return [min(box_x), min(box_y), max(box_x), max(box_y)]


def fourPointTransform(image, rect):
    # errCode = ErrorCode.SUCCESS
    rect = ( [int(rect[0][0]), int(rect[0][1])], [int(rect[1][0]), int(rect[1][1])], [int(rect[2][0]), int(rect[2][1])], [int(rect[3][0]), int(rect[3][1])]  )
    (tl, tr, br, bl) = rect

    widthA = numpy.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = numpy.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = numpy.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = numpy.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # print("heightAB", heightA, heightB)
    maxHeight = max(int(heightA), int(heightB))

    rect = numpy.float32(rect)
    dst = numpy.array([
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # return the warped image
    return  warped

def sorted_box(box_list):

    #box_1 = topleft, box3 = topright, box4 = b_right, box2 = b_left
    box_list = sorted(box_list, key = lambda box_list: box_list[0])
    box_12 = box_list[:2]
    box_34 = box_list[2:]

    box_12 = sorted(box_12, key = lambda box_12: box_12[1])
    box1 = box_12[0]
    box2 = box_12[1]

    box_34 = sorted(box_34, key = lambda box_34: box_34[1])
    box3 = box_34[0]
    box4 = box_34[1]

    return [box1, box3, box4, box2]

def check_box(box_list, box_true):
    #clear_box: box= false==> convert to rectangle
    box_list = sorted_box(box_list)

    [box1, box3, box4, box2] = box_list

    #angle_line1(top):
    goc_line1 = (box3[1] - box1[1])/(box3[0] - box1[0])
    #angle_line2(Buton)
    goc_line2 = (box4[1] - box2[1])/(box4[0] - box2[0])
    #line_3(right):
    goc_line3 = (box4[1] - box3[1])/(box4[0] - box3[0])
    #line4(left):
    goc_line4 = (box2[1] - box1[1])/(box2[0] - box1[0])

    if abs(goc_line1 - goc_line2) >0.12 or abs(goc_line3 - goc_line4)> 0.12:
        return box_true
    else:
        return box_list

def caculator_score_u2net(image, box_max, predict_mask_ori):
    #creat_mask_from four_point
    mask_final =  np.zeros(image.shape, np.uint8)
    mask_final = cv2.drawContours(mask_final , [box_max], -1 ,(255,255,255),-1)
    mask_final = cv2.resize(mask_final, (320,320))
    # mask_final = mask_final /255
    mask_final = Image.fromarray(mask_final)
    mask_final = mask_final.convert('L')
    mask_final = ImageOps.grayscale(mask_final)
    mask_final = np.array(mask_final)
    
    score_list_u2net = []
    # print(mask_final.shape, 'mask.shape=============')
    for x_w in range(320):
        for x_h in range(320):
            if mask_final[x_w][x_h] != 0:
                score_list_u2net.append(predict_mask_ori[x_w][x_h])
    
    score_list_u2net = np.array(score_list_u2net)
    return np.mean(score_list_u2net)

if __name__ == '__main__':

    #link_drive_models: "https://drive.google.com/drive/folders/10pAkqr-ZErb3KTtJtSt7dAoRkDmv_afv"

    #load_model_yolov5
    weight = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/detect_point/cp/best_yolov5_151222_V1.pt" #weights_yolov5
    model, device = load_model(weights=weight)
    print("GPU Memory_____: %s" % getMemoryUsage())

    #load_model_u2net
    net_u2net = U2NETP(3,1)
    THRESHOLD = 0.2
    RESCALE = 255
    THRESHOLD_score_u2net = 0.92
    model_dir = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/detect_point/cp/u2net_p_1612_bess.pth"
    if torch.cuda.is_available():
        net_u2net.load_state_dict(torch.load(model_dir))
        net_u2net.cuda()
    else:        
        net_u2net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    net_u2net.eval()

    #foler_save_output
    folder_output = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/TV_L-20221208T075619Z-001/check_croped_141222_v3/"

    #load_path_img_input
    data_folder = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/panasonic-20221209T082601Z-001/panasonic/"
    with open( data_folder +  'paths.txt','r') as f:
        IMAGE_PATH_DB = [line.strip('\n') for line in f.readlines()]

    err = 0
    total = 0
    for path in IMAGE_PATH_DB:
        total += 1
        img_ori = cv2.imread(data_folder + path)

        center = img_ori.shape
        h0, w0 = center[0], center[1]
        start = timeit.default_timer()
        box_img, box_image_no = detect_box(model, device, img_ori,imgsz=[640,640],conf_thres=0.3, iou_thres = 0.3)
        # print("box_image_no", box_image_no)
        img_output = folder_output + path

        if len(box_image_no) == 4:
            box_crop = []
            for box in box_image_no:
                b = [box[0], box[1]]
                box_crop.append(b)
            # box_crop = check_box(box_crop, None)
            box_crop = sorted_box(box_crop)
            if box_crop != None and 0.66 <box_crop[0][1] / box_crop[1][1] < 1.5 and 0.7 < box_crop[0][0] / box_crop[3][0] < 1.5 and 0.7 < box_crop[1][0] / box_crop[2][0] < 1.5 and 0.7 < box_crop[1][0] / box_crop[2][0] < 1.5 and 0.7 < box_crop[2][1] / box_crop[3][1] < 1.4 : 
                box_crop = np.array(box_crop)
                box_crop = np.int0(box_crop) 
                box_output = box_crop
                # img_final = cv2.drawContours(img_ori.copy(),[box_crop],0,(0,0,256),2)
                # img_final = fourPointTransform(img_ori.copy(), box_crop )
            else:
                img_final = img_ori.copy()
                image = io.imread(data_folder + path)
                image_pil = Image.fromarray(image)
                w, h = image.shape[1],image.shape[0]
                # print("shape", image.shape)
                mask_unet, predict_mask_ori = predict(image, net_u2net, THRESHOLD, RESCALE )
                box_mers, results, box_max = getBoxPanels(mask_unet)
                if len(box_mers) == 1:
                    box_max = sorted_box(box_max)
                    box_max = np.int0(box_max)
                    score_u2net = caculator_score_u2net(image, box_max, predict_mask_ori)
                    if score_u2net > THRESHOLD_score_u2net:
                        box_output = box_max
                    # img_final = fourPointTransform(img_ori.copy(), box_max )
                    # img_final = cv2.drawContours(img_final,[box_max],0,(0,255,256),2)
                    else:
                        box_output = []
                else:
                    box_output = []
        else:
            img_final = img_ori.copy()
            image = io.imread(data_folder + path)
            image_pil = Image.fromarray(image)
            w, h = image.shape[1],image.shape[0]
            # print("shape", image.shape)
            mask_unet, predict_mask_ori = predict(image, net_u2net, THRESHOLD, RESCALE )
            box_mers, results, box_max = getBoxPanels(mask_unet)
            if len(box_mers) ==1:
                box_max = sorted_box(box_max)
                box_max = np.int0(box_max)
                score_u2net = caculator_score_u2net(image, box_max, predict_mask_ori)\

                if score_u2net > THRESHOLD_score_u2net:
                    box_output = box_max
                else:
                    box_output = []
                # img_final = cv2.drawContours(img_final,[box_max],0,(0,255,256),2)
                # img_final = fourPointTransform(img_ori.copy(), box_max)
            else:
                box_output = []
        stop = timeit.default_timer()
        print('Time: ', stop - start)  
        # print(box_output, "boxoutput")
        if len(box_output) == 4:
            img_final = fourPointTransform(img_ori.copy(), box_output )
            cv2.imwrite(img_output, img_final)
        else:
            err+=1
    #err = number image no four point, total = total image
    print(err, total, err/total)