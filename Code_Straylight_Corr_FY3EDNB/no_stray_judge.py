import numpy as np
import pylab as plt
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
import matplotlib.path as mpath
import cv2
import shutil
from scipy.signal import find_peaks,peak_prominences,peak_widths
###outside modules
from resize_pict import resize_image


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def origin_pict_path(date,outdir):
    date_dir=os.path.join(outdir,date)
    filldark_name=date+'_filldark.png'
    if filldark_name in os.listdir(date_dir):
        suffix='_filldark.png'
    else:
        suffix='.png'
    pict_path=os.path.join(date_dir,date+suffix)
    
    return pict_path,date_dir


def resize_origin_pict(date,outdir):
    date_dir=os.path.join(outdir,date)
    origin_pict_path=os.path.join(date_dir,date+'.png')
    ######resize the image to (2000,1522)
    resize_image(origin_pict_path)
    filldark_name=date+'_filldark.png'
    if filldark_name in os.listdir(date_dir):
        pict_path=os.path.join(date_dir,filldark_name)
        ######resize the image to (2000,1522)
        resize_image(pict_path)
    

def if_stray_light(date,outdir):
    stray_light=1
    part_stray_light=0
    pict_path,date_dir=origin_pict_path(date,outdir)
    src = cv2.imread(pict_path)
   
    ####这里再将原图resize the image to (2000,1522)  ####Here resize the original image to (2000,1522)
    resize_origin_pict(date,outdir)
    
    gray=cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) #将彩色图像转换为灰度图  #Convert color image to grayscale
    hist, _ = np.histogram(gray.flatten(),bins=255, range=[0, 254])
    ####暗的判断  ####Dark judgment
    proportion_0=round(hist[0]/sum(hist),2)
    if proportion_0>=0.85:  ###hist0值比例要大于等于0.85  ###The proportion of hist0 values should be greater than or equal to 0.85
        stray_light=0
        # plt.plot(hist)
        # plt.ylim(0,200000)
        # plt.text(185,180000,'P0={}'.format(proportion_0),fontsize=16,fontweight='bold')
        # plt.savefig(os.path.join(date_dir,date+'peak.png'),
        #                                 dpi=300,bbox_inches='tight')
        # plt.close('all')
    
    ####亮的判断  ####Bright judgment
    if stray_light==1:
        smoothed_hist= moving_average(hist, window_size=20)  ###原来是20  ###Originally 20
        peaks, _ = find_peaks(smoothed_hist, distance=100,height=9000,prominence=4000)
        prominences = peak_prominences(smoothed_hist, peaks)[0]
        widths, h_eval, left_ips, right_ips = peak_widths(smoothed_hist, peaks, rel_height=0.95,)
        num_peaks = len(peaks)
        if num_peaks==1:
            left_len=round(peaks[0]-left_ips[0],2)
            right_len=round(right_ips[0]-peaks[0],2)
            ##### 找到局部波峰波谷 ,最终确定用显著度15000  #####Find local peaks and valleys, finally determined using prominence 15000
            # 找到波峰  #Find peaks
            crests, _ = find_peaks(hist,prominence=15000)
            # 找到波谷  #Find valleys
            valleys, _ = find_peaks(-hist,prominence=15000)
            crest_and_valley=len(crests)+len(valleys)
                
            """此处为显著度大于等于80000的"""  """This is for prominence greater than or equal to 80000"""
            if (prominences[0]>=80000)&(right_len<90):  ####分段判断亮的无杂散光的条件  ####Segmented judgment conditions for bright without stray light
                if peaks[0]>=110:
                    """画直方图部分，可省略"""  """Histogram plotting part, can be omitted"""
                    # plt.plot(hist)
                    # p0=plt.plot(smoothed_hist)
                    # plt.hlines(h_eval, left_ips, right_ips, color="green",lw=2)  ####画图要放到if之后  ####Drawing should be placed after if
                    # plt.ylim(0,200000)
                    # for j in range(len(peaks)):
                    #     plt.axvline(x=peaks[j],linestyle='--',c='r')
                    # plt.text(185,180000,'left={}'.format(round(peaks[0]-left_ips[0],2)),fontsize=13)
                    # plt.text(185,165000,'right={}'.format(round(right_ips[0]-peaks[0],2)),fontsize=13)
                    # # plt.savefig(os.path.join(date_dir,date+'peak.png'),
                    # #                             dpi=300,bbox_inches='tight')
                    # plt.close('all')
                    
                    stray_light=0
                else:
                    peaks, _ = find_peaks(smoothed_hist, distance=60,height=9000,prominence=3400) 
                    num_peaks = len(peaks)
                    loc_maxpeak=np.argmax(prominences)
                    if (num_peaks==1)or((loc_maxpeak==1)&(num_peaks==2)):  ####可转换为and的转入部分杂散光处理条件  ####Can be converted to conditions for partial stray light processing with 'and'
                        """画直方图部分，可省略"""  """Histogram plotting part, can be omitted"""
                        # plt.plot(hist)
                        # p0=plt.plot(smoothed_hist)
                        # plt.hlines(h_eval, left_ips, right_ips, color="green",lw=2)  ####画图要放到if之后  ####Drawing should be placed after if
                        # plt.ylim(0,200000)
                        # for j in range(len(peaks)):
                        #     plt.axvline(x=peaks[j],linestyle='--',c='r')
                        # plt.text(185,180000,'left={}'.format(round(peaks[0]-left_ips[0],2)),fontsize=13)
                        # plt.text(185,165000,'right={}'.format(round(right_ips[0]-peaks[0],2)),fontsize=13)
                        # plt.savefig(os.path.join(date_dir,date+'peak.png'),
                        #                             dpi=300,bbox_inches='tight')
                        # plt.close('all')
                    
                        stray_light=0
                    else:
                        part_stray_light=1
                
            """此处为显著度小于80000的"""  """This is for prominence less than 80000"""
            if (prominences[0]<80000)&(crest_and_valley>=10)&(right_len<90):
                peaks, _ = find_peaks(smoothed_hist, distance=60,height=9000,prominence=3400) 
                num_peaks_2=len(peaks)
    
                if (num_peaks_2==1)or((num_peaks_2>1)&(crest_and_valley>=40)):
                    # plt.plot(hist)
                    # p0=plt.plot(smoothed_hist)
                    # plt.hlines(h_eval, left_ips, right_ips, color="green",lw=2)  ####画图要放到if之后  ####Drawing should be placed after if
                    # plt.ylim(0,200000)
                    # for j in range(len(peaks)):
                    #         plt.axvline(x=peaks[j],linestyle='--',c='r')
                    # plt.text(185,180000,'left={}'.format(round(peaks[0]-left_ips[0],2)),fontsize=13)
                    # plt.text(185,165000,'right={}'.format(round(right_ips[0]-peaks[0],2)),fontsize=13)
                    # plt.savefig(os.path.join(date_dir,date+'peak.png'),
                    #                                 dpi=400,bbox_inches='tight')
                    # plt.close('all')
                    
                    stray_light=0
                    
                if 10 in peaks:
                    num_peaks_2=num_peaks_2-1
                if (num_peaks==2)&(crest_and_valley<40):  ####这三行转入部分杂散光处理条件  ####These three lines transfer to partial stray light processing conditions
                    part_stray_light=1
                    
    ####这里接下来继续判断是否为部分杂散光  ####Here continue to judge whether it is partial stray light
    if part_stray_light==0:
        """阈值横向分割后计算两部分的平均差值"""  """Calculate the average difference between two parts after threshold horizontal segmentation"""
        # print('theshold desicion')
        ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU  #Method selected as THRESH_OTSU
        ll_mean=np.nanmean(gray,axis=1)       
        row=np.argmin(abs(ll_mean-ret1))    
        mean1=np.mean(gray[row+1:,:])
        mean2=np.mean(gray[:row+1,:])
        cha=round(abs(mean1-mean2),2)
        
        if cha>90:
            part_stray_light=1
                    
    return pict_path,date_dir,stray_light,part_stray_light


