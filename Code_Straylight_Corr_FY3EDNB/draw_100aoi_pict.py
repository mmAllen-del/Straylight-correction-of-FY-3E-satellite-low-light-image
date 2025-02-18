import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.path as mpath
from copy import copy
import cv2



def draw_fy3e_100aoi(rad_file,ll,lons,lats,angle,outdir,dpi0):
     
    #####此处判断是否有缺值，避免重复画图  #####Check for missing values here to avoid redrawing
    loc_0=np.where((ll==0.0)&(angle>=100))
    if len(loc_0[0])>0:
        dark=1
    else:
        dark=0
 
    
    """对白天进行赋nan值"""  """Assign nan values for daytime"""
    loc_day=np.where(angle<100)
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

    # print(ll_min,ll_max)
    year = int(rad_file[-26:-22])
    month = int(rad_file[-22:-20])
    day = int(rad_file[-20:-18])
    hour = int(rad_file[-17:-15])
    minute = int(rad_file[-15:-13])
    
    dt = datetime.datetime(year, month, day, hour, minute, 0)
    time = dt.strftime("%Y%m%d %H:%M UTC")
    
    
    ####画图程序  ####Drawing program
    fig = plt.figure(figsize=(6,10))
    ax = plt.axes()
    my_cmap = copy(plt.cm.get_cmap('gray'))
    my_cmap.set_under('black')
    norm0=mpl.colors.LogNorm(vmin=ll_min, vmax=ll_max)
    img1=ax.pcolormesh(ll, cmap=my_cmap,norm= norm0)
    plt.xticks([])
    plt.yticks([])

    # 把边框弄掉  # Remove the border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # output as .png
    outname =  dt.strftime("%Y%m%d_%H%M") + '.png'
    date=dt.strftime("%Y%m%d_%H%M")
    # makedir date
    folder_path=os.path.join(outdir,date)
    if  os.path.exists(folder_path):
        None
    else:
        os.makedirs(folder_path)
    outpath = os.path.join(outdir,date,outname)
    fig.savefig(outpath, format='png', dpi=dpi0, bbox_inches='tight',pad_inches=0)
    plt.close('all')
   
    return dark,norm0




def fill_missing_values(data):
    # 获取数据的行数和列数  # Get the number of rows and columns of the data
    rows, cols = data.shape
    # 创建一个新的数组，以便不在原数组上操作  # Create a new array to avoid operating on the original array
    new_data = data.copy()

    # 定义一个辅助函数来获取周围的值并计算平均  # Define a helper function to get surrounding values and calculate average
    def get_average(x, y):
        max_distance = max(rows, cols)  # 最大可能的搜索半径  # Maximum possible search radius
        for size in range(1, max_distance):
            x_min = max(x - size, 0)
            x_max = min(x + size + 1, rows)
            y_min = max(y - size, 0)
            y_max = min(y + size + 1, cols)
            sub_array = new_data[x_min:x_max, y_min:y_max]
            valid_values = sub_array[~np.isnan(sub_array)]
            if valid_values.size > 0:
                return np.mean(valid_values)
        return np.nan  # 如果整个矩阵都是缺失值，返回nan  # If the entire matrix contains missing values, return nan

    # 遍历数组寻找缺失值  # Traverse the array to find missing values
    for i in range(rows):
        for j in range(cols):
            if np.isnan(data[i, j]):
                # 动态扩大查找窗口直到找到有效值  # Dynamically expand search window until valid values are found
                avg = get_average(i, j)
                new_data[i, j] = avg

    return new_data


def draw_fy3e_100aoi_filldark(rad_file,ll,lons,lats,angle,outdir,dpi0):
    
    # """增添夜间的0值处理"""  """Add processing for nighttime 0 values"""
    loc_0=np.where((ll==0.0)&(angle>=100))
    ll[loc_0]=np.nan
    filled_data = fill_missing_values(ll)
    filled_dark_ll=filled_data.copy()
    
    """填补完再对白天进行赋nan值"""  """After filling, assign nan values for daytime"""
    loc_day=np.where(angle<100)
    filled_data[loc_day]=np.nan
    
    loc=np.where((filled_data>0)&(filled_data<=90)&(-180<lons)&(lons<180)&(-90.<lats)&(lats<90.))
    ll_min=np.min(filled_data[loc])
    ll_max=np.max(filled_data[loc])
    
    if ll_max>=3*10**(-5):
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

    
    year = int(rad_file[-26:-22])
    month = int(rad_file[-22:-20])
    day = int(rad_file[-20:-18])
    hour = int(rad_file[-17:-15])
    minute = int(rad_file[-15:-13])
    
    dt = datetime.datetime(year, month, day, hour, minute, 0)
    time = dt.strftime("%Y%m%d %H:%M UTC")
    
    
    ####画图程序  ####Drawing program
    fig = plt.figure(figsize=(6, 10))
    ax = plt.axes()
    my_cmap = copy(plt.cm.get_cmap('gray'))
    my_cmap.set_under('black')
    norm0=mpl.colors.LogNorm(vmin=ll_min, vmax=ll_max)
    img1=ax.pcolormesh(filled_data , cmap=my_cmap,norm= norm0)
    plt.xticks([])
    plt.yticks([])

    # 把边框弄掉  # Remove the border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # output as .png
    outname =  dt.strftime("%Y%m%d_%H%M") + '_filldark.png'
    date=dt.strftime("%Y%m%d_%H%M")
    # makedir date
    folder_path=os.path.join(outdir,date)
    outpath = os.path.join(outdir,date,outname)
    fig.savefig(outpath, format='png', dpi=dpi0, bbox_inches='tight',pad_inches=0)
    plt.close('all')
    return filled_dark_ll

