
"""对不同类型杂散光图像的处理步骤"""
"""Processing steps for different types of stray light images"""

from algorithm_details import *
from PIL import Image
from draw_geo_pict import *

def no_stray_light_handle(pict_filled,date_dir,date,f,ll_output):   ##21-94
    src = cv2.imread(pict_filled)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)   ##将彩色图像转换为灰度图  ##Convert color image to grayscale
    
    padding = np.zeros((2000, 7))
    ll_output_1=np.hstack((padding, ll_output, padding))
    f.create_dataset('No_stray_light_image', data=ll_output_1.astype('float32'))
    f['No_stray_light_image'].attrs['image_type'] = 'no stray light images'
    f['No_stray_light_image'].attrs['intercept'] = 0.0
    f['No_stray_light_image'].attrs['slope'] = 1.0
    f['No_stray_light_image'].attrs['long_name'] = 'Low light band radiance data after filling 0 values'
    f['No_stray_light_image'].attrs['units'] = 'W/m2/sr'
    f['No_stray_light_image'].attrs['valid_range'] = (0,200)
    
    nostripes_pict=enhance_and_destripe(gray)
    
    save_path=os.path.join(date_dir,date+'_correction.png')
    cv2.imwrite(save_path,nostripes_pict)
    ###写入hdf5  ###Write to HDF5
    nostripes_pict=cv2.imread(save_path)
    nostripes_pict = cv2.cvtColor(nostripes_pict, cv2.COLOR_BGR2GRAY)
    
    nostripes_pict_1=np.hstack((padding, nostripes_pict, padding))
    f.create_dataset('Enhanced_image', data=np.flipud(nostripes_pict_1))

    f['Enhanced_image'].attrs['image_type'] = 'no stray light images'
    f['Enhanced_image'].attrs['intercept'] = 0.0
    f['Enhanced_image'].attrs['slope'] = 1.0
    f['Enhanced_image'].attrs['long_name'] = 'gray-scale values of the image after enhancement'
    f['Enhanced_image'].attrs['units'] = 'None'
    f['Enhanced_image'].attrs['valid_range'] = (0, 255)
    
    f.close()

    

def part_stray_light_handle(pict_filled,date_dir,date,rad_file,origin_ll,lons,lats,angle,dpi0,f,draw_flag):  ##98-
    tem_path=date_dir
    
    src1 = cv2.imread(pict_filled)
    gray_img=cv2.cvtColor(src1, cv2.COLOR_RGB2GRAY)  ##将彩色图像转换为灰度图  ##Convert color image to grayscale
    ret1, th1 = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU  #Method selected as THRESH_OTSU
    ll_mean=np.nanmean(gray_img,axis=1)       
    row=np.argmin(abs(ll_mean-ret1))
    row_poportion=round(row*1.0/gray_img.shape[0],3)
    dividing_line=gray_img[row,:]
    med=np.median(dividing_line)
    rows=np.shape(gray_img)[0]*1.0
    zhanbi=min(row/rows,(rows-row)/rows)
    if round(zhanbi,3)<0.2:     ###设定0.2的占比为阈值，小于0.2直接进行亮度统一  ###Set 0.2 as threshold, directly unify brightness if less than 0.2
        # print('part threshold < 0.2!!')
        brightness_img=unify_brightness(gray_img, med)
        if draw_flag==2:
            save_path=os.path.join(date_dir,date+'_brightness.png')
            cv2.imwrite(save_path,brightness_img)
        ####写入hdf5  ####Write to HDF5
        padding = np.zeros((2000, 7))
        brightness_img_1=np.hstack((padding, brightness_img, padding))
        f.create_dataset('No_stray_light_image', data=np.flipud(brightness_img_1))
        f['No_stray_light_image'].attrs['image_type'] = 'partial stray light images'
        f['No_stray_light_image'].attrs['intercept'] = 0.0
        f['No_stray_light_image'].attrs['slope'] = 1.0
        f['No_stray_light_image'].attrs['long_name'] = 'gray-scale values of the image after stray light correction'
        f['No_stray_light_image'].attrs['units'] = 'None'
        f['No_stray_light_image'].attrs['valid_range'] = (0, 255)
        
        
    else:  ###大于等于0.2的进行分区去雾  ###For values greater than or equal to 0.2, perform regional dehazing
        # print('part threshold > 0.2!!')
        mean1,mean2=np.nanmean(gray_img[row+1:,:]),np.nanmean(gray_img[:row+1,:])
        if mean1>mean2:
            bright_flag='after'
        else:
            bright_flag='before'
        gauss_fn=gauss_fitting(pict_filled,date, rad_file,origin_ll,lons,lats,angle,tem_path,dpi0,bright_flag,row_poportion) ###从这里调guass拟合的函数，生成图片后再读取  ###Call Gaussian fitting function here, generate image then read
        src = cv2.imread(gauss_fn)
            
        I = src.astype('float64')/255
        I_origin=I.copy()
    

        #####判断两端亮度后将两边图像处理后拼接起来  #####After judging brightness at both ends, process and stitch images together
        if mean1>mean2:
            ####给row+1行以前的部分去雾  ####Dehaze the part before row+1
            for channel in range(3): 
                src[:row+1,:,channel]=255
            I = src.astype('float64')/255
            dark = DarkChannel(I,15)
            A = AtmLight(I,dark)
            te = TransmissionEstimate(I,A,15,0.8)    ####用0.85和0.7  ####Use 0.85 and 0.7
            t = TransmissionRefine(src,te)
            J = Recover(I,t,A,0.1)
            
            for channel in range(3):
                J[:row+1,:,channel]=I_origin[:row+1,:,channel]
        else:
             ####给row+1行及以后的部分去雾  ####Dehaze the part from row+1 onwards
            for channel in range(3):
                src[row+1:,:,channel]=255
            I = src.astype('float64')/255
            dark = DarkChannel(I_origin,15)
            A = AtmLight(I_origin,dark)
            te = TransmissionEstimate(I_origin,A,15,0.8)
            t = TransmissionRefine(src,te)
            J = Recover(I_origin,t,A,0.1)
            
            for channel in range(3):
                J[row+1:,:,channel]=I_origin[row+1:,:,channel]
        
        dehaze_img=J*255

        dehaze_img_gray=cv2.cvtColor(dehaze_img.astype('uint8'), cv2.COLOR_RGB2GRAY)
        
        brightness_img=unify_brightness(dehaze_img_gray, med)
        if draw_flag==2:
            cv2.imwrite(os.path.join(tem_path,date+'_dehaze.png'),dehaze_img)
            save_path=os.path.join(date_dir,date+'_brightness.png')
            cv2.imwrite(save_path,brightness_img)
            
        ####写入hdf5  ####Write to HDF5
        padding = np.zeros((2000, 7))
        brightness_img_1=np.hstack((padding, brightness_img, padding))
        f.create_dataset('No_stray_light_image', data=np.flipud(brightness_img_1))
        f['No_stray_light_image'].attrs['image_type'] = 'partial stray light images'
        f['No_stray_light_image'].attrs['intercept'] = 0.0
        f['No_stray_light_image'].attrs['slope'] = 1.0
        f['No_stray_light_image'].attrs['long_name'] = 'gray-scale values of the image after stray light correction'
        f['No_stray_light_image'].attrs['units'] = 'None'
        f['No_stray_light_image'].attrs['valid_range'] = (0, 255)
        
        
        
    nostripes_pict=enhance_and_destripe(brightness_img)
    # if (nostripes_pict.shape[0]!=2000) or (nostripes_pict.shape[1]!=1522):
    #     nostripes_pict = cv2.resize(nostripes_pict , (1522, 2000), interpolation=cv2.INTER_LINEAR)
    
    save_path=os.path.join(date_dir,date+'_correction.png')
    cv2.imwrite(save_path,nostripes_pict)
    ####写入hdf5  ####Write to HDF5
    nostripes_pict=cv2.imread(save_path)
    nostripes_pict = cv2.cvtColor(nostripes_pict, cv2.COLOR_BGR2GRAY)
    
    
    nostripes_pict_1=np.hstack((padding, nostripes_pict, padding))
    f.create_dataset('Enhanced_image', data=np.flipud(nostripes_pict_1))
    
    f['Enhanced_image'].attrs['image_type'] = 'partial stray light images'
    f['Enhanced_image'].attrs['intercept'] = 0.0
    f['Enhanced_image'].attrs['slope'] = 1.0
    f['Enhanced_image'].attrs['long_name'] = 'gray-scale values of the image after enhancement'
    f['Enhanced_image'].attrs['units'] = 'None'
    f['Enhanced_image'].attrs['valid_range'] = (0, 255)
    f.close()

        
    
    
def common_stray_light_handle(pict_filled,date_dir,date,rad_file,ll,lons,lats,solarzenith,dpi0,pict_norm,f,draw_flag):
    tem_path=date_dir
    
    gauss_fn=gauss_fitting_common_2(pict_filled,date,rad_file,ll,lons,lats,solarzenith,tem_path,dpi0,pict_norm)
    src = cv2.imread(gauss_fn)
    dst = adaptive_light_correction_black(src)
    dst_gray=cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    if draw_flag==2:
        save_path=os.path.join(date_dir,date+'_brightness.png')
        cv2.imwrite(save_path,dst_gray)
        
    padding = np.zeros((2000, 7))
    dst_gray_1=np.hstack((padding, dst_gray, padding))
    f.create_dataset('No_stray_light_image', data=np.flipud(dst_gray_1))
    f['No_stray_light_image'].attrs['image_type'] = 'common stray light images'
    f['No_stray_light_image'].attrs['intercept'] = 0.0
    f['No_stray_light_image'].attrs['slope'] = 1.0
    f['No_stray_light_image'].attrs['long_name'] = 'gray-scale values of the image after stray light correction'
    f['No_stray_light_image'].attrs['units'] = 'None'
    f['No_stray_light_image'].attrs['valid_range'] = (0, 255)
    
    ####经过实验，不需要再进行去雾算法，亮度均匀后直接增强效果更好  ####Through experiments, dehazing algorithm is not needed, better effect by directly enhancing after brightness uniformity
    dehaze_img=cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)   
    nostripes_pict=enhance_and_destripe(dehaze_img)
    
    save_path=os.path.join(date_dir,date+'_correction.png')
    cv2.imwrite(save_path,nostripes_pict)
    ####写入hdf5  ####Write to HDF5
    nostripes_pict=cv2.imread(save_path)
    nostripes_pict = cv2.cvtColor(nostripes_pict, cv2.COLOR_BGR2GRAY)
    
    nostripes_pict_1=np.hstack((padding, nostripes_pict, padding))
    f.create_dataset('Enhanced_image', data=np.flipud(nostripes_pict_1))
    
    f['Enhanced_image'].attrs['image_type'] = 'common stray light images'
    f['Enhanced_image'].attrs['intercept'] = 0.0
    f['Enhanced_image'].attrs['slope'] = 1.0
    f['Enhanced_image'].attrs['long_name'] = 'gray-scale values of the image after enhancement'
    f['Enhanced_image'].attrs['units'] = 'None'
    f['Enhanced_image'].attrs['valid_range'] = (0, 255)
    f.close()

    
def handle_remove_pict(hdf5_file,date_dir,date,lons,lats,angle,ll,stray_light,dpi0,draw_flag):
    if draw_flag!=2:
        os.remove(os.path.join(date_dir,date+'.png'))
        os.remove(os.path.join(date_dir,date+'_correction.png'))
        try:
            os.remove(os.path.join(date_dir,date+'_filldark.png'))
        except:
            None
        try:
            os.remove(os.path.join(date_dir,date+'_gauss.png'))
        except:
            None
    if draw_flag==1:
        draw_fy3e_ll(lons,lats,angle,ll,date_dir,date,dpi0)
        if stray_light==0:
            draw_hdf5_geo_image_no_stray_light(hdf5_file, date_dir,date,dpi0)
            draw_hdf5_geo_image(hdf5_file, date_dir,date,2,dpi0)
        else:
            choice=1          #######选择无杂散图像(1)还是增强图像(2)  #######Choose no stray light image (1) or enhanced image (2)
            draw_hdf5_geo_image(hdf5_file,date_dir,date,choice,dpi0)
            choice=2          
            draw_hdf5_geo_image(hdf5_file,date_dir,date,choice,dpi0)
