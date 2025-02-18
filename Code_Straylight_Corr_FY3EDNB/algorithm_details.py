
"""对杂散光图像的处理算法"""
"""Processing algorithms for stray light images"""

"""指引-函数功能依次为1.固有的增强和去条纹算法
                    2.部分杂散光的亮度统一
                    3.去雾算法（针对部分杂散）
                    4.普通杂散光图像的不均匀光照订正"""
"""Guide - Functions in order: 1. Inherent enhancement and destriping algorithm
                            2. Brightness unification for partial stray light
                            3. Dehazing algorithm (for partial stray light)
                            4. Non-uniform illumination correction for common stray light images"""


import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import os
import numpy as np
import pandas as pd
import time
import re
import math
import datetime
import matplotlib.colors as colors
from copy import copy
import matplotlib.path as mpath
import cv2
import pywt
from scipy.optimize import curve_fit
###outside modules
from gauss_fitting import *

###############以下是固有的增强和去条纹算法########################
###############The following is the inherent enhancement and destriping algorithm########################
def apply_clahe(gray):
    # 将图像转换为灰度图像
    # Convert image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 创建CLAHE对象
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # 应用自适应直方图均衡化
    # Apply adaptive histogram equalization
    equalized_image = clahe.apply(gray)
    
    # 返回光照归一化后的图像
    # Return the normalized image
    return equalized_image


def damp_coefficient(coeff, sigma):
    """Filter DWT coefficients by performing an FFT and applying a Gaussian
    kernel.
    """
    fft_coeff = np.fft.fft(coeff, axis=0)
    fft_coeff = np.fft.fftshift(fft_coeff, axes=[0])

    ydim, _ = fft_coeff.shape
    gauss1d = 1 - np.exp(-np.arange(-ydim // 2, ydim // 2)**2 / (2 * sigma**2))
    damped_fc = fft_coeff * gauss1d[:, np.newaxis]

    damped_coeff = np.fft.ifftshift(damped_fc, axes=[0])
    damped_coeff = np.fft.ifft(damped_coeff, axis=0)
    return damped_coeff.real

def remove_stripes(image, decomp_level, wavelet, sigma):
    """Removes stripes from `image` with a combined wavelet/FFT approach.

    Params
    ------
    image : 2d array
        containing the stripy image
    decomp_level : int
        Decomposition level of DWT (TODO: could be automatically calculated?)
    wavelet : str
        name of wavelet to use for DWT
    sigma : int
        sigma of Gaussian that is used to smooth FFT coefficients
    """
    coeffs = pywt.wavedec2(
        image, wavelet=wavelet, level=decomp_level, mode='symmetric')

    damped_coeffs = [coeffs[0]]

    for ii in range(1, len(coeffs)):
        ch, cv, cd = coeffs[ii]

        cv = damp_coefficient(cv, sigma)
        ch = damp_coefficient(ch, sigma)

        damped_coeffs.append((ch, cv, cd))

    rec_image = pywt.waverec2(damped_coeffs, wavelet=wavelet, mode='symmetric')
    return rec_image


def picture_stripes(gray_img):
    rotated_img = cv2.rotate(gray_img, cv2.ROTATE_90_CLOCKWISE) ####向右旋转90度  ####Rotate 90 degrees to the right
    nostripes = remove_stripes(rotated_img,5, 'bior5.5', 10)
    nostripes_r=cv2.rotate(nostripes, cv2.ROTATE_90_COUNTERCLOCKWISE) ####向左旋转回来90度  ####Rotate 90 degrees back to the left
    return nostripes_r


def enhance_and_destripe(image):
    normalized_image = apply_clahe(image)
    nostripes=picture_stripes(normalized_image)
    return nostripes

###############固有的增强和去条纹算法########################
###############Inherent enhancement and destriping algorithm########################



"""最终用分割线的中值法确定了亮度统一"""
"""Finally determined brightness unification using the median method of dividing lines"""
def unify_brightness(gray_img,median):
    for i in range(np.shape(gray_img)[0]):
        other_line=np.array(gray_img[i,:],dtype='float32')
        med_1=np.median(other_line)
        cha=med_1-median
        if median<80:
            other1=other_line-cha
        elif median<120:
            other1=other_line-cha-30
        else:
            other1=other_line-cha-50
        loc=np.where(other1<0)
        other1[loc]=0
        other1=np.array(other1,dtype='uint8')
        gray_img[i,:]=other1
    
    return gray_img





###############去雾算法########################
###############Dehazing algorithm########################
def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark
 
def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz,1)
    imvec = im.reshape(imsz,3)
 
    indices = darkvec.argsort()
    indices = indices[imsz-numpx::]
 
    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]
 
    A = atmsum / numpx;
    return A
 
def TransmissionEstimate(im,A,sz,omega):
    # omega = 0.5,0.95
    im3 = np.empty(im.shape,im.dtype)
 
    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission
 
def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p
 
    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I
 
    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I
 
    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))
 
    q = mean_a*im + mean_b
    return q
 
def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)
 
    return t
 
def Recover(im,t,A,tx):
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t,tx)
 
    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]
 
    return res

###############去雾算法########################
###############Dehazing algorithm########################



##############普通杂散光图像的不均匀光照订正#################
##############Non-uniform illumination correction for common stray light images#################
"""基于二维伽马函数的光照不均匀图像自适应校正算法(去除白色区域)"""
"""Adaptive correction algorithm for non-uniform illumination images based on 2D gamma function (removing white areas)"""

def adaptive_light_correction_black(img):
    height = img.shape[0]
    width = img.shape[1]
    
    loc=np.where(img==255)
    img[loc]=np.mean(img[np.where(img<255)])
    
    HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    V= HSV_img[:,:,2]
    
    
    kernel_size = min(height, width)
    if kernel_size %2 == 0:
        kernel_size -= 1
    
    SIGMA1 = 15
    SIGMA2 = 80
    SIGMA3 = 250
    q= np.sqrt(2.0)
    
    F= np.zeros((height,width,3),dtype=np.float64)
    F[:,:,0]= cv2.GaussianBlur(V,(kernel_size, kernel_size),SIGMA1 / q)
    F[:,:,1]= cv2.GaussianBlur(V,(kernel_size, kernel_size),SIGMA2 / q)
    F[:,:,2]= cv2.GaussianBlur(V,(kernel_size, kernel_size),SIGMA3 / q) 
    
    
    F_mean = np.mean(F,axis=2)
    average =np.mean(F_mean)
    gamma = np.power(0.5,np.divide(np.subtract(average, F_mean),average))
    out = np.power(V/255.0,gamma)*255.0
    HSV_img[:,:,2]= out
    img = cv2.cvtColor(HSV_img, cv2.COLOR_HSV2BGR)
    img[loc]=255
    
    return img

##############普通杂散光图像的不均匀光照订正#################
##############Non-uniform illumination correction for common stray light images#################

