
"""指引-函数功能依次为1.针对部分杂散光的高斯拟合
                    2.普通杂散光图像的高斯拟合
                    3.针对部分杂散光的拟合后画图
                    4.普通杂散光图像的拟合后画图
Guide - Functions are:
1. Gaussian fitting for partial stray light
2. Gaussian fitting for normal stray light images  
3. Plot after fitting for partial stray light
4. Plot after fitting for normal stray light images"""

import numpy as np
import pylab as plt
import matplotlib.cm as cm
import h5py
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from copy import copy
import matplotlib as mpl
import datetime
import time
import os
###outside modules
from resize_pict import resize_image



# 创建一个函数模型用来生成数据 Create a function model to generate data
def gaussian(x, a, x0, y0, sigma_x, sigma_y):
    r=a*np.exp(-((x[0]-x0)**2/(2*sigma_x**2) + (x[1]-y0)**2/(2*sigma_y**2)))
    return r.ravel()

def fill_missing_values(data):
    # 获取数据的行数和列数 Get number of rows and columns
    rows, cols = data.shape
    # 创建一个新的数组，以便不在原数组上操作 Create a new array to avoid operating on original array
    new_data = data.copy()

    # 定义一个辅助函数来获取周围的值并计算平均 Define helper function to get surrounding values and calculate average
    def get_average(x, y):
        max_distance = max(rows, cols)  # 最大可能的搜索半径 Maximum possible search radius
        for size in range(1, max_distance):
            x_min = max(x - size, 0)
            x_max = min(x + size + 1, rows)
            y_min = max(y - size, 0)
            y_max = min(y + size + 1, cols)
            sub_array = new_data[x_min:x_max, y_min:y_max]
            valid_values = sub_array[~np.isnan(sub_array)]
            if valid_values.size > 0:
                return np.mean(valid_values)
        return np.nan  # 如果整个矩阵都是缺失值，返回nan Return nan if entire matrix has missing values

    # 遍历数组寻找缺失值 Traverse array to find missing values
    for i in range(rows):
        for j in range(cols):
            if np.isnan(data[i, j]):
                # 动态扩大查找窗口直到找到有效值 Dynamically expand search window until valid values are found
                avg = get_average(i, j)
                new_data[i, j] = avg

    return new_data

##############针对部分杂散光的高斯拟合 Gaussian fitting for partial stray light####################
def gauss_fitting(pict_filled,date,rad_file,ll,lons,lats,solarzenith,tem_path,dpi0,bright_flag,row_poportion):
    try:
        
        ####----------实际操作可不要 Not required for actual operation--------####
        # """增添夜间的0值处理 Add processing for nighttime 0 values"""
        loc_0=np.where((ll==0.0)&(solarzenith>=100))
        ll[loc_0]=np.nan
        filled_data = fill_missing_values(ll)
        """填补完再对白天进行赋nan值 After filling, assign nan values for daytime"""
        ll=filled_data.copy()
        ####----------实际操作可不要 Not required for actual operation--------####
        
        loc_day=np.where((solarzenith<100))
        ll[loc_day]=np.nan
        loc=np.where((ll>0)&(ll<90)&(-180<lons)&(lons<180)&(-90.<lats)&(lats<90.))
        
    

        ######去除完全没有杂散光的区域 Remove areas with no stray light
        ll_no=ll.copy()
        threshold=np.percentile(ll_no[loc], 30)
        loc=np.where((ll>0)&(ll<90)&(-180<lons)&(lons<180)&(-90.<lats)&(lats<90.)&(ll_no>threshold))
        #####去除背景值后拟合杂散光 Fit stray light after removing background values
        ll_min=np.min(ll[loc])
        ll_lim=np.percentile(ll[loc], 80)
        
        ll_1,ll_correct=ll.copy(),ll.copy()
        loc_not_stray=np.where((ll_1<=ll_lim)|(ll_1>90))
        ll_1[loc_not_stray]=np.nan
        
        ll_fit=ll_1-ll_lim
        loc_nan=np.where(np.isnan(ll_fit))
        ll_fit[loc_nan]=0
        loc_stray=np.where(ll_fit>0)       ####用于拟合的杂散光数据位置 Location of stray light data for fitting
        # 生成原始数据 Generate original data
        x1 = np.linspace(0, 41, 1522).reshape(1, -1)
        x2 = np.linspace(0, 56, 2000).reshape(1, -1)
        X, Y = np.meshgrid(x1, x2)
        
        XX = np.expand_dims(X, 0)
        YY = np.expand_dims(Y, 0)
        xx = np.append(XX, YY, axis=0)

        ###使用curve_fit函数拟合噪声数据 Use curve_fit function to fit noisy data
        popt, pcov = curve_fit(gaussian, xx,ll_fit.ravel(),maxfev=500000)
        #####画图 Plot********************
        my_cmap = copy(plt.cm.get_cmap('gray'))
        my_cmap.set_under('black')
        ###尝试订正 Attempt correction
        R = gaussian(xx, *popt)
        R = R.reshape(2000, 1522)
        
        if bright_flag=='after':
            row=int((1-row_poportion)*2000)
            ll_correct[:row+1,:]=ll_correct[:row+1,:]-R[:row+1,:]
        else:
            row=int(row_poportion*2000)
            ll_correct[row+1:,:]=ll_correct[row+1:,:]-R[row+1:,:]
    
        ll[loc_day],ll_correct[loc_day]=np.nan,np.nan
        outpath=draw_part_gauss_100aoi(ll_correct,rad_file,tem_path,dpi0)
        return outpath
    except:
        print(pict_filled,'too long time!!!!')
        return pict_filled

##############针对部分杂散光的高斯拟合 Gaussian fitting for partial stray light####################



##############普通杂散光图像的高斯拟合 Gaussian fitting for normal stray light images####################

def gauss_fitting_common_2(pict_filled,date,rad_file,ll,lons,lats,solarzenith,tem_path,dpi0,norm0):
    try:
        ####需要读取原辐射值来拟合 Need to read original radiation values for fitting
        loc_day=np.where((solarzenith<100))
        ll[loc_day]=np.nan
        loc=np.where((ll>0)&(ll<90)&(-180<lons)&(lons<180)&(-90.<lats)&(lats<90.))

        #####去除背景值后拟合杂散光 Fit stray light after removing background values
        ll_lim=np.percentile(ll[loc], 50)
        ll_1,ll_correct=ll.copy(),ll.copy()
        loc_not_stray=np.where(ll_1<=ll_lim)
        ll_1[loc_not_stray]=np.nan
        
        ll_fit=ll_1-ll_lim
        loc_nan=np.where(np.isnan(ll_fit)|(ll_1>90))
        ll_fit[loc_nan]=0
        # 生成原始数据 Generate original data
        x1 = np.linspace(0, 41, 1522).reshape(1, -1)
        x2 = np.linspace(0, 56, 2000).reshape(1, -1)
        X, Y = np.meshgrid(x1, x2)

        XX = np.expand_dims(X, 0)
        YY = np.expand_dims(Y, 0)
        xx = np.append(XX, YY, axis=0)

        try:
            ###使用curve_fit函数拟合噪声数据 Use curve_fit function to fit noisy data
            popt, pcov = curve_fit(gaussian, xx,ll_fit.ravel(),maxfev=500)
            ############检测是否有负值，若负值较多则转变成百分位90 Check for negative values, if too many then switch to 90th percentile
            R = gaussian(xx, *popt)
            R = R.reshape(2000, 1522)
            loc_negative=np.where(ll_correct<R)
            if len(loc_negative[0])>10000:
                # print('negative_num=',len(loc_negative[0]))
                try:
                    lim_90=np.percentile(ll[loc], 90)
                    ll_1=ll.copy()
                    loc_not_fit=np.where(ll_1>=lim_90)
                    loc_50=np.where((ll>ll_lim)&(ll<lim_90))
                    ll_1[loc_not_fit]=np.nanmean(ll_1[loc_50])
                    
                    ll_fit=ll_1-ll_lim
                    loc_nan=np.where(np.isnan(ll_fit)|(ll_fit>90))
                    ll_fit[loc_nan]=0
                    ###使用curve_fit函数拟合噪声数据 Use curve_fit function to fit noisy data
                    popt, pcov = curve_fit(gaussian, xx,ll_fit.ravel(),maxfev=500)
                except:
                    ll_lim=np.percentile(ll[loc], 90)
                    ll_1=ll.copy()
                    loc_not_stray=np.where(ll_1<=ll_lim)
                    ll_1[loc_not_stray]=np.nan

                    ll_fit=ll_1-ll_lim
                    loc_nan=np.where(np.isnan(ll_fit)|(ll_fit>90))
                    ll_fit[loc_nan]=0
                    popt, pcov = curve_fit(gaussian, xx, ll_fit.ravel(), maxfev=500)

            
        ############迭代时间较长，对背景阈值逐步提高 Long iteration time, gradually increase background threshold
        except:
            try:
                lim_90=np.percentile(ll[loc], 70)
                ll_1=ll.copy()
                loc_not_stray=np.where(ll_1<=ll_lim)
                ll_1[loc_not_stray]=np.nan
                ll_fit=ll_1-lim_90
                loc_nan=np.where(np.isnan(ll_fit)|(ll_fit>90))
                ll_fit[loc_nan]=0
                popt, pcov = curve_fit(gaussian, xx, ll_fit.ravel(), maxfev=500)
            except:
                try:
                    lim_90=np.percentile(ll[loc], 80)
                    ll_1=ll.copy()
                    loc_not_stray=np.where(ll_1<=ll_lim)
                    ll_1[loc_not_stray]=np.nan
                    ll_fit=ll_1-lim_90
                    loc_nan=np.where(np.isnan(ll_fit)|(ll_fit>90))
                    ll_fit[loc_nan]=0
                    popt, pcov = curve_fit(gaussian, xx, ll_fit.ravel(), maxfev=500)
                except:
                    lim_90=np.percentile(ll[loc], 90)
                    try:
                        ll_1=ll.copy()
                        loc_not_stray=np.where(ll_1<=ll_lim)
                        ll_1[loc_not_stray]=np.nan
                        ll_fit=ll_1-lim_90
                        loc_nan=np.where(np.isnan(ll_fit)|(ll_fit>90))
                        ll_fit[loc_nan]=0
                        popt, pcov = curve_fit(gaussian, xx, ll_fit.ravel(), maxfev=500)
                    except:
                        ll_1=ll.copy()
                        loc_not_fit=np.where(ll_1>=lim_90)
                        loc_50=np.where((ll>ll_lim)&(ll<lim_90))
                        ll_1[loc_not_fit]=np.nanmean(ll_1[loc_50])
                        
                        ll_fit=ll_1-ll_lim
                        loc_nan=np.where(np.isnan(ll_fit)|(ll_fit>90))
                        ll_fit[loc_nan]=0
                        ###使用curve_fit函数拟合噪声数据 Use curve_fit function to fit noisy data
                        popt, pcov = curve_fit(gaussian, xx,ll_fit.ravel(),maxfev=500)
                        
                
            
            
            # # # popt返回最拟合给定的函数模型func的参数值 popt returns parameter values that best fit the given function model func
            # print('1--',popt)
        #####画图 Plot********************
        my_cmap = copy(plt.cm.get_cmap('gray'))
        my_cmap.set_under('black')
        ###尝试订正 Attempt correction
        R = gaussian(xx, *popt)
        R = R.reshape(2000, 1522)
        ll_correct=ll_correct-R
        ll[loc_day],ll_correct[loc_day]=np.nan,np.nan
        outpath=draw_common_gauss_100aoi(ll_correct,rad_file,tem_path,dpi0,norm0)
        return outpath
    except:
        print(pict_filled,'too long time!!!!')
        return pict_filled

##############普通杂散光图像的高斯拟合 Gaussian fitting for normal stray light images####################

"""含有拟合的时间 Contains fitting time"""
def gauss_fitting_common_1(pict_filled,date,rad_file,geo_file,tem_path,dpi0,norm0):
        ####需要读取原辐射值来拟合 Need to read original radiation values for fitting
        dts_l1b = h5py.File(rad_file, 'r')
        ll = np.array(dts_l1b['Data/EV_1KM_LL'])
        dts_geo = h5py.File(geo_file, 'r')
        lons = np.array(dts_geo['Geolocation/Longitude'])
        lats = np.array(dts_geo['Geolocation/Latitude'])
        solarzenith=np.array(dts_geo['Geolocation']['SolarZenith'][:])*0.01
        dts_l1b.close()
        dts_geo.close()
        
        array_0=[0,1,2,3,4,5,6,1529,1530,1531,1532,1533,1534,1535]
        ll = np.delete(ll,array_0, axis=1)
        lons = np.delete(lons, array_0, axis=1)
        lats= np.delete(lats, array_0, axis=1)
        solarzenith= np.delete(solarzenith, array_0, axis=1)
        loc_day=np.where((solarzenith<100))
        ll[loc_day]=np.nan
        loc=np.where((ll>0)&(ll<90)&(-180<lons)&(lons<180)&(-90.<lats)&(lats<90.))

        #####去除背景值后拟合杂散光 Fit stray light after removing background values
        ll_lim=np.percentile(ll[loc], 90)
        ll_1,ll_correct=ll.copy(),ll.copy()
        loc_not_stray=np.where(ll_1<=ll_lim)
        ll_1[loc_not_stray]=np.nan
        
        ll_fit=ll_1-ll_lim
        loc_nan=np.where(np.isnan(ll_fit)|(ll_1>90))
        ll_fit[loc_nan]=0
        loc_stray=np.where(ll_fit>0)
        # 生成原始数据 Generate original data
        x1 = np.linspace(0, 41, 1522).reshape(1, -1)
        x2 = np.linspace(0, 56, 2000).reshape(1, -1)
        X, Y = np.meshgrid(x1, x2)

        XX = np.expand_dims(X, 0)
        YY = np.expand_dims(Y, 0)
        xx = np.append(XX, YY, axis=0)


        ###使用curve_fit函数拟合噪声数据 Use curve_fit function to fit noisy data
        time0=time.time()
        popt, pcov = curve_fit(gaussian, xx,ll_fit.ravel(),maxfev=500)
        time1=time.time()
        print('the fitting use {}s'.format(round(time1-time0),2))
        print(popt)
        #####画图 Plot********************
        my_cmap = copy(plt.cm.get_cmap('gray'))
        my_cmap.set_under('black')
        ###尝试订正 Attempt correction
        R = gaussian(xx, *popt)
        R = R.reshape(2000, 1522)
        ll_correct=ll_correct-R
        ll[loc_day],ll_correct[loc_day]=np.nan,np.nan
        outpath=draw_common_gauss_100aoi(ll_correct,rad_file,tem_path,dpi0,norm0)
        return outpath



def gauss_fitting_1(pict_filled,date,rad_file,geo_file,tem_path,dpi0,bright_flag,row_poportion):
    ####需要读取原辐射值来拟合 Need to read original radiation values for fitting
    dts_l1b = h5py.File(rad_file, 'r')
    ll = np.array(dts_l1b['Data/EV_1KM_LL'])
    dts_geo = h5py.File(geo_file, 'r')
    lons = np.array(dts_geo['Geolocation/Longitude'])
    lats = np.array(dts_geo['Geolocation/Latitude'])
    solarzenith=np.array(dts_geo['Geolocation']['SolarZenith'][:])*0.01
    dts_l1b.close()
    dts_geo.close()
    
    array_0=[0,1,2,3,4,5,6,1529,1530,1531,1532,1533,1534,1535]
    ll = np.delete(ll,array_0, axis=1)
    lons = np.delete(lons, array_0, axis=1)
    lats= np.delete(lats, array_0, axis=1)
    solarzenith= np.delete(solarzenith, array_0, axis=1)
    loc_day=np.where((solarzenith<100))
    ll[loc_day]=np.nan
    loc=np.where((ll>0)&(ll<90)&(-180<lons)&(lons<180)&(-90.<lats)&(lats<90.))

    ######去除完全没有杂散光的区域 Remove areas with no stray light
    ll_no=ll.copy()
    threshold=np.percentile(ll_no[loc], 30)
    loc=np.where((ll>0)&(ll<90)&(-180<lons)&(lons<180)&(-90.<lats)&(lats<90.)&(ll_no>threshold))
    #####去除背景值后拟合杂散光 Fit stray light after removing background values
    ll_min=np.min(ll[loc])
    ll_lim=np.percentile(ll[loc], 80)
    
    ll_1,ll_correct=ll.copy(),ll.copy()
    loc_not_stray=np.where((ll_1<=ll_lim)|(ll_1>90))
    ll_1[loc_not_stray]=np.nan
    
    ll_fit=ll_1-ll_lim
    loc_nan=np.where(np.isnan(ll_fit))
    ll_fit[loc_nan]=0
    loc_stray=np.where(ll_fit>0)       ####用于拟合的杂散光数据位置 Location of stray light data for fitting
    # 生成原始数据 Generate original data
    x1 = np.linspace(0, 41, 1522).reshape(1, -1)
    x2 = np.linspace(0, 56, 2000).reshape(1, -1)
    X, Y = np.meshgrid(x1, x2)
    
    XX = np.expand_dims(X, 0)
    YY = np.expand_dims(Y, 0)
    xx = np.append(XX, YY, axis=0)

    ###使用curve_fit函数拟合噪声数据 Use curve_fit function to fit noisy data
    time0=time.time()
    popt, pcov = curve_fit(gaussian, xx,ll_fit.ravel(),maxfev=500000)
    time1=time.time()
    print('the fitting use {}s'.format(round(time1-time0),2))
    #####画图 Plot********************
    my_cmap = copy(plt.cm.get_cmap('gray'))
    my_cmap.set_under('black')
    ###尝试订正 Attempt correction
    R = gaussian(xx, *popt)
    R = R.reshape(2000, 1522)
    
    if bright_flag=='after':
        row=int((1-row_poportion)*2000)
        ll_correct[:row+1,:]=ll_correct[:row+1,:]-R[:row+1,:]
    else:
        row=int(row_poportion*2000)
        ll_correct[row+1:,:]=ll_correct[row+1:,:]-R[row+1:,:]
   
    ll[loc_day],ll_correct[loc_day]=np.nan,np.nan
    outpath=draw_part_gauss_100aoi(ll_correct,rad_file,tem_path,dpi0)
    return outpath



"""--------------以下是画图部分 Below is the plotting section-----------------------------"""

##############普通杂散光图像的高斯拟合画图 Plot Gaussian fitting for normal stray light images####################
def draw_common_gauss_100aoi(ll,rad_file,outdir,dpi0,norm0):
    year = int(rad_file[-26:-22])
    month = int(rad_file[-22:-20])
    day = int(rad_file[-20:-18])
    hour = int(rad_file[-17:-15])
    minute = int(rad_file[-15:-13])
    
    dt = datetime.datetime(year, month, day, hour, minute, 0)
    time = dt.strftime("%Y%m%d %H:%M UTC")
    
    ####画图程序 Plotting program
    fig = plt.figure(figsize=(6,8))
    ax = plt.axes()
    my_cmap =  mpl.cm.get_cmap("gray").copy()
    my_cmap.set_under('black')
    img1=ax.pcolormesh(ll , cmap =my_cmap,norm= norm0)
    plt.xticks([])
    plt.yticks([])

    # 把边框弄掉 Remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # output as .png
    outname =  dt.strftime("%Y%m%d_%H%M") + '_gauss.png'
    date=dt.strftime("%Y%m%d")
    outpath = os.path.join(outdir,outname)
    fig.savefig(outpath, format='png', dpi=dpi0, bbox_inches='tight',pad_inches=0)
    plt.close('all')
    ######resize the image to (2000,1522)
    resize_image(outpath)
    return outpath


##############针对部分杂散光的高斯拟合画图 Plot Gaussian fitting for partial stray light####################
def draw_part_gauss_100aoi(ll,rad_file,outdir,dpi0):
    loc=np.where(ll>0)
    ll_min=np.min(ll[loc])
    ll_max=np.max(ll[loc])

    if ll_max>=3*10**(-5):
        if ll_min<3*10**(-5):
            ll_min=3*10**(-5)
        
    if (ll_max<2*10**(-3)):
         ll_max=1*10**(-3)
    if (ll_max>2*10**(-3))&(ll_max<3*10**(-3)):
        ll_max=2*10**(-3)
    if (ll_max>3*10**(-3))&(ll_max<4*10**(-3)):
        ll_max=3*10**(-3)
    if (ll_max>4*10**(-3))&(ll_max<5*10**(-3)):
        ll_max=4*10**(-3)
    if (ll_max>=5*10**(-3)):
        ll_max=5*10**(-3)

    
    year = int(rad_file[-26:-22])
    month = int(rad_file[-22:-20])
    day = int(rad_file[-20:-18])
    hour = int(rad_file[-17:-15])
    minute = int(rad_file[-15:-13])
    
    dt = datetime.datetime(year, month, day, hour, minute, 0)
    time = dt.strftime("%Y%m%d %H:%M UTC")
    
    ####画图程序 Plotting program
    fig = plt.figure(figsize=(6,8))
    ax = plt.axes()
    my_cmap =  mpl.cm.get_cmap("gray").copy()
    my_cmap.set_under('black')
    img1=ax.pcolormesh(ll , cmap =my_cmap,norm= mpl.colors.LogNorm(vmin=ll_min, vmax=ll_max))
    plt.xticks([])
    plt.yticks([])

    # 把边框弄掉 Remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # output as .png
    outname =  dt.strftime("%Y%m%d_%H%M") + '_gauss.png'
    date=dt.strftime("%Y%m%d")
    outpath = os.path.join(outdir,outname)
    fig.savefig(outpath, format='png', dpi=dpi0, bbox_inches='tight',pad_inches=0)
    plt.close('all')
    ######resize the image to (2000,1522)
    resize_image(outpath)
    return outpath


