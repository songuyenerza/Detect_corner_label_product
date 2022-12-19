u2net_path = "U-2-Net"
from distutils.log import error
import sys
from tkinter import W, Image
from turtle import window_height
import cv2
import os
import numpy as np
sys.path.append(u2net_path)
from PIL import Image, ImageOps
from model import U2NETP
from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
import timeit

transform =transforms.Compose([RescaleT(320),
                            ToTensorLab(flag=0)])



def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn
def predict(image, net, THRESHOLD, RESCALE ):
    image_cp = image.copy()
    h, w = image.shape[1],image.shape[0]
    image = transform(image)
    inputs_test = image.type(torch.FloatTensor)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)
    inputs_test = inputs_test.unsqueeze(0)
    d1,d2,d3,d4,d5,d6,d7= net(inputs_test)
    pred = d1[:,0,:,:]
    predict = normPRED(pred)
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    
    predict_mask_ori = predict_np.copy()  #mask_original

    for a in range(predict_np.shape[0]):
        for b in range(predict_np.shape[1]):
            if predict_np[a][b] >THRESHOLD:
                predict_np[a][b] = 1
            if predict_np[a][b] <= THRESHOLD:
                predict_np[a][b] = 0

    # rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # predict_np = cv2.dilate(predict_np, rect_kernel, iterations = 1)
    # predict_np = cv2.erode(predict_np, rect_kernel, iterations = 3)

    # kernel = np.ones((7,7),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    mask = cv2.erode(predict_np,kernel,iterations = 4)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    predict_np = cv2.dilate(mask,kernel,iterations = 1)


    alpha = Image.fromarray(predict_np*255)
    alpha = np.array(alpha).astype(np.uint8)

    drawing = Image.fromarray(alpha)
    drawing = drawing.convert('L')
    drawing = drawing.resize((h,w), resample=Image.BILINEAR)
    drawing = ImageOps.grayscale(drawing)
    drawing = np.array(drawing)
    return  drawing , predict_mask_ori


def mer_box(boxs):
    box0 = []
    box1 = []
    box2=[]
    box3=[]

    for box in boxs:
        # print("boxx",box)
        for b in box:
            box0.append(b[0])
            box1.append(b[1])
            # box2.append(box[2][0])
            # box3.append(box[2][1])
    x0 = min(box0)
    y0 = min(box1)
    x1 = max(box0)
    y1 = max(box1)
    
    bounding_box = [int(x0) , int (y0) , int(x1), int(y1)]
    return bounding_box
# def getBoxPanels(mask):
#     a = np.where(mask != 0)
#     boxes = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
#     return boxes

def sorted_box(box_list, w0, h0):

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
    # print("check====================", [box1, box3, box4, box2], w0, h0)

    if box1[0]< 3 or box1[1] < 3 or box2[0] < 3 or w0 - box3[0] < 3 or w0 - box4[0] < 3 or box2[1] < 3 or h0 - box4[1] < 3 or h0 - box2[1] < 3:
        return [] 
    else:
        return [box1, box3, box4, box2]

# def check_box(box_list, box_true):
#     #clear_box: box= false==> convert to rectangle
#     box_list = sorted_box(box_list)

#     [box1, box3, box4, box2] = box_list

#     #angle_line1(top):
#     if box3[0] - box1[0] != 0:
#         goc_line1 = (box3[1] - box1[1])/(box3[0] - box1[0])
#     else:
#         goc_line1 = 0
#     #angle_line2(Buton)
#     if box4[0] - box2[0] != 0:
#         goc_line2 = (box4[1] - box2[1])/(box4[0] - box2[0])
#     else:
#         goc_line2 = 0
#     #line_3(right):
#     if box4[0] - box3[0] != 0:
#         goc_line3 = (box4[1] - box3[1])/(box4[0] - box3[0])
#     else:
#         goc_line3 = 0
#     #line4(left):
#     if box2[0] - box1[0] != 0:
#         goc_line4 = (box2[1] - box1[1])/(box2[0] - box1[0])
#     else:
#         goc_line4 = 0

#     if abs(goc_line1 - goc_line2) >0.5 or abs(goc_line3 - goc_line4)> 0.5:
#         return box_true
#     else:
#         return box_list


def getBoxPanels(mask):
    h0, w0 = mask.shape
    dilation = Image.fromarray(mask)
    dilation = ImageOps.grayscale(dilation)
    dilation = np.array(dilation)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    results = []
    area_list = []
    approx_list = []
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06* perimeter, True)
        # print('approx', approx)
        approx_list.append(approx)
        area = cv2.contourArea(c)
        # if area < 0.05*h0*w0 :
        #     continue
        rect = cv2.minAreaRect(c)
        x, y, w, h = cv2.boundingRect(c)
        # boxes.append((x, y ,x +w , y + h))
        results.append([x, y ,x +w , y + h])
        area_list.append(area)
        box = cv2.boxPoints(rect)
        # box = np.int0(box)
        boxes.append(box)
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # exit()	
    if len(area_list) > 0:
        index = np.argmax(area_list)
        boxes_max = boxes[index]
        if len(approx_list[index]) ==4:
            box_max = approx_list[index]
            box_max = [box_max[0][0], box_max[1][0], box_max[2][0], box_max[3][0]]

            # box_max = check_box(box_max, boxes_max)
            box_max = sorted_box(box_max, w0, h0)
            box_max = np.array(box_max)
            box_max = np.int0(box_max)

        else:
            box_max = boxes_max
            box_max = sorted_box(box_max,  w0, h0)
            box_max = np.array(box_max)
            box_max = np.int0(box_max)
    else:
        box_max = []

    return boxes, results, box_max

if __name__ == "__main__":
    net = U2NETP(3,1)
    THRESHOLD = 0.9
    RESCALE = 255
    model_dir = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/detect_point/cp/u2net_p_1612_bess.pth"
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:        
        net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    net.eval()

    data_folder_input = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/panasonic-20221209T082601Z-001/panasonic/"
    data_folder_output = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/TV_L-20221208T075619Z-001/checku2net/"


    with open( data_folder_input +  'paths.txt','r') as f:
        IMAGE_PATH_DB = [line.strip('\n') for line in f.readlines()]

    for path in IMAGE_PATH_DB:
        start = timeit.default_timer()
        
        img_path = data_folder_input + path
        img_save = data_folder_output + path
        image = io.imread(img_path)
        # image_pil = Image.open(img_path)
        image_pil = Image.fromarray(image)

        w, h = image.shape[1],image.shape[0]
        print("shape", image.shape)
        alpha, predict_mask_ori = predict(image, net, THRESHOLD, RESCALE )
        print("predict_mask_ori" , predict_mask_ori, np.mean(predict_mask_ori), predict_mask_ori.shape)
        alpha = Image.fromarray(alpha)

        stop = timeit.default_timer()
        print('Time: ', stop - start) 

        back = Image.new('RGB', (w,h), (255,255,255))
        # print("alpha",alpha)
        print("image_pil",image_pil)
        back.paste(image_pil, (0,0), alpha)
        back.save(img_save, quality=100)
       


