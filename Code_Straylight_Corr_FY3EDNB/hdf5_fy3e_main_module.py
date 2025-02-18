"""
创建于 2024.05.07
版本 2.0
@程序  : 风云三号E星夜间微光图像杂散光校正与增强算法
@作者  : 梁泳恩、闵敏
@邮箱  : vingliangyongen@163.com, minm5@mail.sysu.edu.cn

Created on 2024.05.07
Version 2.0
@Code  : Stray-light Correction and Enhancement Algorithm for Fengyun-3E Satellite Nocturnal Low-light Imagery
@Author: Liang Yongen and Min Min
@E-mail: vingliangyongen@163.com, minm5@mail.sysu.edu.cn
@Reference: Liang Yongen, Min Min(Cor-Author), Hanlie Xu, et al., 2024. Stray light  
            correction and enhancement of nocturnal low-light image of  
            early-morning-orbiting Fengyun-3E satellite [J]. IEEE Transactions 
            on Geoscience and Remote Sensing, 62: 4113113, doi: 10.1109/TGRS.2024.3502441

#  --------------------------------------------------------------------------
# |  Copyright ©, Min Min and Yongen Liang, 2024                             |
# |                                                                          |
# |  All rights reserved. This source code implements the stray-light        |
# |  correction and enhancement algorithm for Fengyun-3E satellite           |
# |  nocturnal low-light imagery, developed for scientific research.         |
# |                                                                          |
# |  The authors (Min and Liang) grant users permission to download,         |
# |  install, and utilize this software solely for scientific research       |
# |  purposes. Redistribution is permitted provided this copyright notice    |
# |  remains intact and proper attribution is given to the authors.          |
# |                                                                          |
# |  This software and any derivative works may not be incorporated into     |
# |  proprietary or commercial software products for sale.                   |
# |                                                                          |
# |  This software is provided "as is" without any express or implied        |
# |  warranties.                                                             |
#  --------------------------------------------------------------------------
"""

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
###outside modules  ###外部模块 
from draw_100aoi_pict import *
from no_stray_judge import *
from stray_light_handle import *

def if_handle_new(rad_file,geo_file):
    flag=1
    dts_l1b = h5py.File(rad_file, 'r')
    ll = np.array(dts_l1b['Data/EV_1KM_LL'])
    ll=ll.astype('float16')
    dts_geo = h5py.File(geo_file, 'r')
    lons = np.array(dts_geo['Geolocation/Longitude'])
    lats = np.array(dts_geo['Geolocation/Latitude'])
    sza=np.array(dts_geo['Geolocation']['SolarZenith'][:])
    angle=np.array(dts_geo['Geolocation']['SolarZenith'][:])*0.01
    dts_l1b.close()
    dts_geo.close()
    ###去除0值的列后矩阵大小为2000*1522  ###Matrix size is 2000*1522 after removing columns with 0 values
    array_0=[0,1,2,3,4,5,6,1529,1530,1531,1532,1533,1534,1535]
    ll = np.delete(ll,array_0, axis=1)
    lons = np.delete(lons, array_0, axis=1)
    lats= np.delete(lats, array_0, axis=1)
    angle= np.delete(angle, array_0, axis=1)
    sza=np.delete(sza, array_0, axis=1)
    origin_ll=ll.copy()
    
    loc_night=np.where(angle>=100)
    night_poportion=round(len(loc_night[0])/(2000*1522),2)
    if night_poportion<0.3:
        print('night poportion less 0.3 !no handle!')
        flag=0

    if abs(night_poportion-1)<10**(-8):  ###全黑夜  ###Complete night
        loc_night_0=np.where(ll==0)
        night_poportion_0=round(len(loc_night_0[0])/(2000*1522),2)
        if night_poportion_0>0.2:
            print('too much 0 in all dark pict !no handle!')
            flag=0
    return flag, origin_ll,lons,lats,sza,angle,night_poportion
    
    


def if_handle(date,outdir,night_poportion):
    flag=1
    origin_pict_path=os.path.join(outdir,date,date+'.png')
    if night_poportion<0.3:
        print('night poportion less 0.3 !no handle!')
        flag=0
    if abs(night_poportion-1)<10**(-8):  ###全黑夜  ###Complete night
        src = cv2.imread(origin_pict_path)
        gray=cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) #将彩色图像转换为灰度图  #Convert color image to grayscale
        loc_255=np.where(gray==255)
        if len(loc_255[0])>0.2*np.size(gray):
            print('too much 0 in all dark pict !no handle!')
            flag=0
    return flag 


###########input date##############
date='20230511_1300'           ###20230511_1300   ;20220613_1620   ;  20220613_1945
rad_file=r'/share/home/dq113/data/1000M/20230511/FY3E_MERSI_GRAN_L1_20230511_1300_1000M_V0.HDF'
geo_file=r'/share/home/dq113/data/GEO1K/20230511/FY3E_MERSI_GRAN_L1_20230511_1300_GEO1K_V0.HDF'

outdir=r'/share/home/dq113/liangye2020/hdf5_final_code_test_v2/outcome_2'
draw_flag=2    #####均生成HDF5,设置画图模式，不画图为0，画地理图为1，画长方形流程图为2  #####All generate HDF5, set drawing mode: 0 for no drawing, 1 for geographic map, 2 for rectangular flow chart
dpi0=300       ###像素控制并统一  ###Pixel control and unification


# print (' +++step 1 Determine whether to handle it+++ ')
flag,origin_ll,lons,lats, solar_zenith_angle,angle,night_poportion=if_handle_new(rad_file,geo_file)


if flag==1:
    # print('***start to handle****')
    # print (' ')
    # print (' +++step 2.1 draw original pictures+++ ')
    darkfill,pict_norm=draw_fy3e_100aoi(rad_file,origin_ll,lons,lats,angle,outdir,dpi0)

    
    # print (' +++step 2.2 whether to draw filled_dark pictures+++ ')
    if darkfill==1:
        filled_dark_ll=draw_fy3e_100aoi_filldark(rad_file,origin_ll,lons,lats,angle,outdir,dpi0)
    else:
        None

    #####这里开始写入hdf5  #####Start writing to hdf5 here
    hdf5_name=os.path.join(outdir,date,'FY3E_MERSI_LL_Clear_Image_'+date+'.hdf5')
    f = h5py.File(hdf5_name,'w')
    
    padding = np.zeros((2000, 7))
    lats_1=np.hstack((padding, lats, padding))
    f.create_dataset('Latitude', data=lats_1.astype('float32'))
    f['Latitude'].attrs['description'] = 'latitude of each image pixel'
    f['Latitude'].attrs['FillValue'] = -9999.9
    f['Latitude'].attrs['intercept'] = 0.0
    f['Latitude'].attrs['slope'] = 1.0
    f['Latitude'].attrs['long_name'] = 'Latitude for each image pixel'
    f['Latitude'].attrs['units'] = 'degree'
    f['Latitude'].attrs['valid_range'] = (-90.0, 90.0)
    
    lons_1=np.hstack((padding, lons, padding))
    f.create_dataset('Longitude', data=lons_1.astype('float32'))
    f['Longitude'].attrs['description'] = 'longitude of each image pixel'
    f['Longitude'].attrs['FillValue'] = -9999.9
    f['Longitude'].attrs['intercept'] = 0.0
    f['Longitude'].attrs['slope'] = 1.0
    f['Longitude'].attrs['long_name'] = 'Longitude for each image pixel'
    f['Longitude'].attrs['units'] = 'degree'
    f['Longitude'].attrs['valid_range'] = (-180.0, 180.0)
    
    solar_zenith_angle_1=np.hstack((padding, solar_zenith_angle, padding))
    f.create_dataset('SolarZenith', data=solar_zenith_angle_1)
    f['SolarZenith'].attrs['description'] = 'solar zenith angle at the center position of each pixel'
    f['SolarZenith'].attrs['FillValue'] = -32767
    f['SolarZenith'].attrs['intercept'] = 0.0
    f['SolarZenith'].attrs['slope'] = 0.01
    f['SolarZenith'].attrs['long_name'] = 'Solar Zenith Angle'
    f['SolarZenith'].attrs['units'] = 'degree'
    f['SolarZenith'].attrs['valid_range'] = (0, 18000)
    
    
    # print (' +++step 2.3 if there is stray light and part stray light+++ ') 
    if abs(night_poportion-1)<10**(-8):      ###必须是全黑夜  ###Must be complete night
        night_poportion=1
        pict_path_filled,date_dir,stray_light,part_stray_light=if_stray_light(date,outdir)
        
        # print('stray_light={},part_stray={}'.format(stray_light,part_stray_light))
        # print (' ')
        # print (' +++step 3 correct the stray light(1)+++ ') 
        ###部分杂散光处理  ###Partial stray light processing
        if part_stray_light==1:
            part_stray_light_handle(pict_path_filled,date_dir,date,rad_file,origin_ll,lons,lats,angle,dpi0,f,draw_flag) 
            
        ###无杂散光处理  ###No stray light processing
        elif stray_light==0:
            if darkfill==1:
                no_stray_light_handle(pict_path_filled,date_dir,date,f,filled_dark_ll)
            else:
                no_stray_light_handle(pict_path_filled,date_dir,date,f,origin_ll)
            
        else:
            ###一般杂散光处理  ###General stray light processing
            common_stray_light_handle(pict_path_filled,date_dir,date,rad_file,origin_ll,lons,lats,angle,dpi0,pict_norm,f,draw_flag)
           
            
        handle_remove_pict(hdf5_name,date_dir,date,lons,lats,angle,origin_ll,stray_light,dpi0,draw_flag)    
    else:
        # print (' ')
        # print (' +++step 3 correct the stray light(2)+++ ') 
        ###一般杂散光处理  ###General stray light processing
        pict_path_filled,date_dir=origin_pict_path(date,outdir)
        resize_origin_pict(date,outdir)
        common_stray_light_handle(pict_path_filled,date_dir,date,rad_file,origin_ll,lons,lats,angle,dpi0,pict_norm,f,draw_flag)
       
        handle_remove_pict(hdf5_name,date_dir,date,lons,lats,angle,origin_ll,1,dpi0,draw_flag)        

    
