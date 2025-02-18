"""
Created on Sun JUN 9 19:58:48 2024

@author: Yongen Liang
"""
"""画论文里带地图的辐射图 Draw radiation maps with geographic features for the paper"""
import matplotlib.pyplot as plt
import h5py
import os
import numpy as np
import pandas as pd
import time
import re
import math
import datetime
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.colors as colors
from copy import copy
import cartopy.feature as cfeature
import matplotlib.path as mpath


def draw_fy3e_ll(lons,lats,angle,ll,date_dir,date,dpi0):
    
    loc_day=np.where(angle<100.)
    loc_night=np.where(angle>=100.)
    ll[loc_day]=np.nan
    loc=np.where((ll>0)&(ll<=90)&(-180<lons)&(lons<180)&(-90.<lats)&(lats<90.))
    ll_min=np.min(ll[loc])
    ll_max=np.max(ll[loc])
    
    if ll_max>3*10**(-5):
        if ll_min<3*10**(-5):
            ll_min=3*10**(-5)
   
    if (ll_max>2*10**(-3))&(ll_max<3*10**(-3)):
        ll_max=2*10**(-3)
    if (ll_max>3*10**(-3))&(ll_max<4*10**(-3)):
        ll_max=3*10**(-3)
    if (ll_max>4*10**(-3))&(ll_max<5*10**(-3)):
        ll_max=4*10**(-3)
    if ll_max>=5*10**(-3):
        ll_max=5*10**(-3)
    
    # print('ll_max=',ll_max,'ll_min=',ll_min)
    
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])
    hour = int(date[9:11])
    minute = int(date[11:13])
    
    dt = datetime.datetime(year, month, day, hour, minute, 0)
    time = dt.strftime("%Y%m%d %H:%M UTC")
    
    
    ###判断是否为极地投影 Determine if polar projection is needed
    polar=0
    lon_180=0
    xmax,xmin=np.max(lons[loc]),np.min(lons[loc])
    ymax,ymin=np.max(lats[loc]),np.min(lats[loc])

    ####判断是否经过180度 Check if crossing 180 degrees longitude
    if (np.floor(xmin) == -180.) and (np.ceil(xmax) == 180.):
        lon_180=1    
        loc_180_1=np.where((ll>0)&(ll<=90)&(0<lons)&(lons<180)&(-90.<lats)&(lats<90.))
        loc_180_2=np.where((ll>0)&(ll<=90)&(-180<lons)&(lons<0)&(-90.<lats)&(lats<90.))
        x_left=np.min(lons[loc_180_1])
        x_right=np.max(lons[loc_180_2])
        xmin=x_left
        xmax=360-abs(x_right)

    loc_70 = np.where((lats[loc] <= 75) & (lats[loc] >= -75))
    size_70=len(loc_70[0])  # fy3e样本点纬度在-75到75的样本数 Number of FY3E samples between -75 and 75 degrees latitude
    size=len(loc[0])
    if (size_70<=0.6*size)or(polar==1):
        print('polar drawing')
        polar=1
        if np.max(lats[loc])<0:
            crs=ccrs.SouthPolarStereo()
            img_extent = [-180,180,-90,ymax]
        else:
            crs=ccrs.NorthPolarStereo()
            img_extent = [-180,180,ymin,90]
    else:
        # print('not polar drawing')
        need=1
        #####判断是否经过180 Check if crossing 180 degrees
        if lon_180==1:
            crs = ccrs.PlateCarree(central_longitude=180)
            # print('180 going*******')
        else:
            crs = ccrs.PlateCarree()
            if ymax-ymin>=2*(xmax-xmin):
                print('need to extend lons********')
                need=0
                if ymax-ymin>3:
                    if xmax+3<180:
                        xmax=xmax+3
                    if xmin-3>-180:
                        xmin=xmin-3
                else:
                    if xmax+1<180:
                        xmax=xmax+1
                    if xmin-1>-180:
                        xmin=xmin-1
            if xmax-xmin>=2*(ymax-ymin) and need==1:
                print('need to extend lats********')
                if ymax+3<90:
                    ymax=ymax+3
                if ymin-3>-90:
                    ymin=ymin-3
        img_extent = [xmin,xmax,ymin,ymax]  # [west,east,south,north]
        
    ####画图程序 Plotting procedure
    fig = plt.figure(figsize=(6, 8))
    ax = plt.axes(projection=crs)
    ###限制范围 Set plot extent
    # print(img_extent)
    ax.set_extent(img_extent,crs = ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor='skyblue', alpha=0.8)  # 添加海洋 Add ocean
    ax.add_feature(cfeature.LAND, facecolor='silver')  # 添加陆地 Add land
    my_cmap = copy(plt.cm.get_cmap('gray'))
    my_cmap.set_under('black')
    
    # img = ax.pcolormesh(lons, lats, ll , cmap =my_cmap,norm=colors.LogNorm(vmax=0.001,vmin=0.0001))
    img = ax.pcolormesh(lons, lats, ll , cmap =my_cmap,norm=colors.LogNorm(vmax=ll_min,vmin=ll_max),
                        transform=ccrs.PlateCarree())
#########ploar:vmax=0.003,vmin=8*10**(-5);
#########not ploar:vmax=0.003,vmin=9.9*10**(-5);

     # # add coastlines and gridlines
    ax.coastlines(resolution='50m',color='green',linewidth=0.5, alpha=0.9)
    
   
    #######以下为网格线的参数 Parameters for gridlines below######
    if polar==1:
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)    #设置axes边界，为圆形边界，否则为正方形的极地投影 Set circular boundary for polar projection, otherwise square

    #######以下为180度的显示刻度问题 Issues with displaying ticks at 180 degrees below#########
    ###中心经度经过180°时以40度为划分以10或5为间隔 When central longitude crosses 180°, use 10 or 5 degree intervals for 40 degree divisions
    if lon_180==1:
        x_locate=[]
        if xmax-xmin>=40:
            interval=10
        else:
            interval=5
        for i in range(int(xmin),int(xmax)):
            if i%interval==0:
                if i>180:
                    x_locate.append(i-360)
                else:
                    x_locate.append(i)
        gl = ax.gridlines(color='gray',linewidth=0.5,linestyle='--', alpha=0.2,
                          draw_labels=True, xlocs=x_locate)
        
    else:
        x_locate=[]
        if xmax-xmin>=40:
            interval=10
        else:
            interval=5
        for i in range(int(xmin),int(xmax)):
            if i%interval==0:
                x_locate.append(i)
        gl = ax.gridlines(color='gray',linewidth=0.5,linestyle='--', alpha=0.2,draw_labels=True,xlocs=x_locate)
    # gl = ax.gridlines(color='gray',linewidth=0.5,linestyle='--', alpha=0.2,draw_labels=True)

    gl.top_labels = False  # 关闭顶端标签 Turn off top labels
    gl.right_labels = False  # 关闭右侧标签 Turn off right labels
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度格式 Set x-axis to longitude format
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度格式 Set y-axis to latitude format
    gl.xlabel_style = {'size': 14, 'color': 'black'}  # 修改label大小 Modify label size
    gl.ylabel_style = {'size': 14, 'color': 'black'}
    
    # add label and title
    plt.title('FY-3E LLB', fontweight='bold', fontsize=16, loc='left')
    plt.title(time, fontsize=16, loc='right')
    
    # # add color bar
    cax = fig.add_axes([ax.get_position().x1+0.05,ax.get_position().y0,0.02,ax.get_position().height])
    cb = plt.colorbar(img, extend='both', orientation='vertical', pad=0.1,fraction=0.035,
                      cax=cax)
    # cb.set_label('W·m$^{-2}$·sr$^{-1}$',size=13)
    cb.ax.tick_params(labelsize=14)
    # cb.ax.set_title('W·m$^{-2}$·sr$^{-1}$',fontsize=12, fontweight='bold')
    
    # output as .png
    # date=dt.strftime("%Y%m%d")
    outname =  dt.strftime("%Y%m%d_%H%M") + '_geo.png'
    outpath = os.path.join(date_dir,outname)
    fig.savefig(outpath, format='png', dpi=dpi0, bbox_inches='tight')
    plt.close('all')

