import matplotlib.pyplot as plt
import re
import numpy as np
import cv2
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# import nibabel as nib
from math import floor
import cmath
from scipy.special import expit
import math
from scipy.ndimage import label
import random
from scipy.ndimage import rotate,find_objects


def align_arrays(arr1, arr2):
    len1 = len(arr1)
    len2 = len(arr2)

    if len1 == len2:
        return arr1, arr2

    min_len = min(len1, len2)
    trim_index = abs(len1 - len2)

    if len1 > len2:
        aligned_arr1 = arr1[trim_index:]
        aligned_arr2 = arr2
    else:
        aligned_arr1 = arr1
        aligned_arr2 = arr2[trim_index:]

    return aligned_arr1, aligned_arr2

def point_transform(points,scale):


    for i in range(len(points)):

            points[i]=points[i]*scale

    return points
# repoints=point_transform(points,[h/256,w/256])
def dist(point1,point2):
    dist=math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return dist
def show_single_point(point, shape):
    mask = np.zeros(shape)
    for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                # print(i ," ",j)
                if dist(point, [i, j]) <= 5:
                    mask[i, j] = 1
    return mask


def line_equation(x1, y1, x2, y2):

    m = -1 *(x2 - x1)/ (y2 - y1) 

    b = y1 - m * x1  

    return m, b

def clip_image(image, x1, y1, x2, y2):
    h, w = image.shape
    result_image = np.zeros_like(image)

    m, b = line_equation(x1, y1, x2, y2)

    for i in range(h):
        for j in range(w):
            if i <= m * j + b:
                result_image[i, j] = image[i, j]

    return result_image


def calculateEF(EDV_key_point,ESV_key_point,Seg_EDV_output_np,Seg_ESV_output_np,h,w):
    #calculate ED 2ch
    EDmask_convert = cv2.resize(Seg_EDV_output_np[0], (int(w[0].item()), int(h[0].item())), interpolation=cv2.INTER_NEAREST)
    ED_points=point_transform(EDV_key_point[0],(h[0].item()/256,w[0].item()/256))
    # ED_points = point_transform(EDV_key_point[0], (1,1))
    mask1p = show_single_point(ED_points[1], EDmask_convert.shape)
    gt_pointed=EDmask_convert/254*0.5+mask1p*0.5
    # plt.subplot(5,1,1)
    # plt.imshow(gt_pointed)
    mid_point_0=(ED_points[0][0] + ED_points[1][0])/2
    mid_point_1 = (ED_points[0][1] + ED_points[1][1])/2
    theta = math.atan((ED_points[2][0] - mid_point_0) / (ED_points[2][1] - mid_point_1))-math.pi/2
    # theta = math.atan((ED_points[0][0] - ED_points[1][0]) / (ED_points[0][1] - ED_points[1][1]))
    # print(theta)
    EDmask_convert=clip_image(image=EDmask_convert,x1=mid_point_1,y1= mid_point_0,x2=ED_points[2][1],y2=ED_points[2][0])

    EDmask_convert = rotate(EDmask_convert/254, theta * 180 / math.pi, reshape=False, order=0)

    # plt.subplot(5, 1, 2)
    # plt.imshow(EDmask_convert)


    axis_length = np.sum(np.max(EDmask_convert, axis=1))
    # objectslices=find_objects(gt_convert)
    # indexs = [i for i in range(EDmask_convert.shape[0]) if np.max(EDmask_convert, axis=1)[i] > 0 and i>ED_points[2][0]]
    indexs = [i for i in range(EDmask_convert.shape[0]) if
              np.max(EDmask_convert, axis=1)[i] > 0 ]
    gt_cropped = EDmask_convert[indexs]
    raidusList2ch = np.sum(gt_cropped, axis=1)
    # print(raidusList4ch)
    # print('L2S ', len(raidusList2ch))

    # plt.subplot(5, 1, 4)
    # plt.imshow(gt_cropped)
    # plt.subplot(5, 1, 5)
    # plt.plot(range(raidusList2ch.shape[0]), raidusList2ch)
    # plt.title('length={}'.format(axis_length))
    # plt.show()

    # calculate ED 4ch
    EDmask_convert = cv2.resize(Seg_EDV_output_np[1], (int(w[1].item()), int(h[1].item())),
                                interpolation=cv2.INTER_NEAREST)
    ED_points = point_transform(EDV_key_point[1], (h[1].item() / 256, w[1].item() / 256))
    # ED_points = point_transform(EDV_key_point[0], (1,1))
    mask1p = show_single_point(ED_points[1], EDmask_convert.shape)
    gt_pointed = EDmask_convert / 254 * 0.5 + mask1p * 0.5
    # plt.subplot(5, 1, 1)
    # plt.imshow(gt_pointed)
    mid_point_0 = (ED_points[0][0] + ED_points[1][0]) / 2
    mid_point_1 = (ED_points[0][1] + ED_points[1][1]) / 2
    theta = math.atan((ED_points[2][0] - mid_point_0) / (ED_points[2][1] - mid_point_1)) - math.pi / 2
    # theta = math.atan((ED_points[0][0] - ED_points[1][0]) / (ED_points[0][1] - ED_points[1][1]))
    # print(theta)
    EDmask_convert=clip_image(image=EDmask_convert,x1=mid_point_1,y1= mid_point_0,x2=ED_points[2][1],y2=ED_points[2][0])
    EDmask_convert = rotate(EDmask_convert / 254, theta * 180 / math.pi, reshape=False, order=0)

    # plt.subplot(5, 1, 2)
    # plt.imshow(EDmask_convert)

    axis_length = np.sum(np.max(EDmask_convert, axis=1))
    # objectslices=find_objects(gt_convert)
    # indexs = [i for i in range(EDmask_convert.shape[0]) if
    #           np.max(EDmask_convert, axis=1)[i] > 0 and i > ED_points[2][0]]
    indexs = [i for i in range(EDmask_convert.shape[0]) if
                       np.max(EDmask_convert, axis=1)[i] > 0 ]
    gt_cropped = EDmask_convert[indexs]
    width = gt_cropped.shape[1]
    # gt_cropped = cv2.resize(gt_cropped, [width, raidusList2ch.shape[0]], interpolation=cv2.INTER_NEAREST)

    raidusList4ch = np.sum(gt_cropped, axis=1)
    # print(raidusList4ch)
    # print('L4S ', len(raidusList4ch))

    # plt.subplot(5, 1, 4)
    # plt.imshow(gt_cropped)
    # plt.subplot(5, 1, 5)
    # plt.plot(range(raidusList2ch.shape[0]), raidusList4ch)
    # plt.title('length={}'.format(axis_length))
    # plt.show()
    raidusList2ch,raidusList4ch = align_arrays(raidusList2ch,raidusList4ch)
    Long_ED=len(raidusList2ch)*0.308/10
    volumeED = 0.308 * math.pi / 4 * (np.sum(raidusList2ch * 0.308 * raidusList4ch * 0.308))/1000
    # print(volumeED)

    # calculate ES 2ch
    ESmask_convert = cv2.resize(Seg_ESV_output_np[0], (int(w[0].item()), int(h[0].item())),
                                interpolation=cv2.INTER_NEAREST)
    ES_points = point_transform(ESV_key_point[0], (h[0].item() / 256, w[0].item() / 256))
    # ED_points = point_transform(EDV_key_point[0], (1,1))
    mask1p = show_single_point(ES_points[1], ESmask_convert.shape)
    gt_pointed = ESmask_convert / 254 * 0.5 + mask1p * 0.5
    # plt.subplot(5, 1, 1)
    # plt.imshow(gt_pointed)
    mid_point_0 = (ES_points[0][0] + ES_points[1][0]) / 2
    mid_point_1 = (ES_points[0][1] + ES_points[1][1]) / 2
    theta = math.atan((ES_points[2][0] - mid_point_0) / (ES_points[2][1] - mid_point_1)) - math.pi / 2
    # theta = math.atan((ES_points[0][0] - ES_points[1][0]) / (ES_points[0][1] - ES_points[1][1]))

    # print(theta)
    ESmask_convert=clip_image(image=ESmask_convert,x1=mid_point_1,y1= mid_point_0,x2=ES_points[2][1],y2=ES_points[2][0])
    ESmask_convert = rotate(ESmask_convert / 254, theta * 180 / math.pi, reshape=False, order=0)

    # plt.subplot(5, 1, 2)
    # plt.imshow(ESmask_convert)

    axis_length = np.sum(np.max(ESmask_convert, axis=1))
    # objectslices=find_objects(gt_convert)
    indexs = [i for i in range(ESmask_convert.shape[0]) if
              np.max(ESmask_convert, axis=1)[i] > 0 ]
    gt_cropped = ESmask_convert[indexs]
    raidusList2ch = np.sum(gt_cropped, axis=1)
    # print(raidusList4ch)
    # print('L2S ', len(raidusList2ch))

    # plt.subplot(5, 1, 4)
    # plt.imshow(gt_cropped)
    # plt.subplot(5, 1, 5)
    # plt.plot(range(raidusList2ch.shape[0]), raidusList2ch)
    # plt.title('length={}'.format(axis_length))
    # plt.show()

    # calculate ES 4ch
    ESmask_convert = cv2.resize(Seg_ESV_output_np[1], (int(w[1].item()), int(h[1].item())),
                                interpolation=cv2.INTER_NEAREST)
    ES_points = point_transform(ESV_key_point[1], (h[1].item() / 256, w[1].item() / 256))
    # ED_points = point_transform(EDV_key_point[0], (1,1))
    mask1p = show_single_point(ES_points[1], ESmask_convert.shape)

    gt_pointed = ESmask_convert / 254 * 0.5 + mask1p * 0.5
    # plt.subplot(5, 1, 1)
    # plt.imshow(gt_pointed)
    mid_point_0 = (ES_points[0][0] + ES_points[1][0]) / 2
    mid_point_1 = (ES_points[0][1] + ES_points[1][1]) / 2
    theta = math.atan((ES_points[2][0] - mid_point_0) / (ES_points[2][1] - mid_point_1)) - math.pi / 2
    # theta = math.atan((ES_points[0][0] - ES_points[1][0]) / (ES_points[0][1] - ES_points[1][1]))
    # print(theta)
    ESmask_convert=clip_image(image=ESmask_convert,x1=mid_point_1,y1= mid_point_0,x2=ES_points[2][1],y2=ES_points[2][0])
    ESmask_convert = rotate(ESmask_convert / 254, theta * 180 / math.pi, reshape=False, order=0)

    # plt.subplot(5, 1, 2)
    # plt.imshow(ESmask_convert)

    axis_length = np.sum(np.max(ESmask_convert, axis=1))
    # objectslices=find_objects(gt_convert)
    indexs = [i for i in range(ESmask_convert.shape[0]) if
              np.max(ESmask_convert, axis=1)[i] > 0 ]
    # print('indexs: ', indexs)
    gt_cropped = ESmask_convert[indexs]
    width = gt_cropped.shape[1]

    raidusList4ch = np.sum(gt_cropped, axis=1)
    raidusList2ch, raidusList4ch = align_arrays(raidusList2ch, raidusList4ch)
    Long_ES=len(raidusList2ch)*0.308/10
    volumeES = 0.308 * math.pi / 4 * (np.sum(raidusList2ch * 0.308 * raidusList4ch * 0.308)) / 1000

    EF=(volumeED-volumeES)/volumeED*100
    return volumeES ,volumeED, EF, Long_ED,Long_ES



