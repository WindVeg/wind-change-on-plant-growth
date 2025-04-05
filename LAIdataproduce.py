# In[] Convert TIFF format to NetCDF format
#  original LAI4g can be downloaded throygh https://zenodo.org/records/8281930

from osgeo import gdal
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import calendar
import pandas as pd
import geopandas as gpd
import salem


def tiff2nc(path):
    data = gdal.Open(path)
    im_width = data.RasterXSize  # 获取宽度，数组第二维，左右方向元素长度，代表经度范围
    im_height = data.RasterYSize  # 获取高度，数组第一维，上下方向元素长度，代表纬度范围
    im_bands = data.RasterCount  # 波段数
    """
    GeoTransform 的含义：
        影像左上角横坐标：im_geotrans[0]，对应经度
        影像左上角纵坐标：im_geotrans[3]，对应纬度

        遥感图像的水平空间分辨率(纬度间隔)：im_geotrans[5]
        遥感图像的垂直空间分辨率(经度间隔)：im_geotrans[1]
        通常水平和垂直分辨率相等

        如果遥感影像方向没有发生旋转，即上北下南，则 im_geotrans[2] 与 im_geotrans[4] 为 0

    计算图像地理坐标：
        若图像中某一点的行数和列数分别为 row 和 column，则该点的地理坐标为：
            经度：xGeo = im_geotrans[0] + col * im_geotrans[1] + row * im_geotrans[2]
            纬度：yGeo = im_geotrans[3] + col * im_geotrans[4] + row * im_geotrans[5]
    """
    im_geotrans = data.GetGeoTransform()  # 获取仿射矩阵，含有 6 个元素的元组

    im_proj = data.GetProjection()  # 获取地理信息

    """
    GetRasterBand(bandNum)，选择要读取的波段数，bandNum 从 1 开始
    ReadAsArray(xoff, yoff, xsize, ysize)，一般就按照下面这么写，偏移量都是 0 ，返回 ndarray 数组
    """
    im_data = data.GetRasterBand(1).ReadAsArray(xoff=0, yoff=0, win_xsize=im_width, win_ysize=im_height)
    # 根据im_proj得到图像的经纬度信息
    im_lon = [im_geotrans[0] + i * im_geotrans[1] for i in range(im_width)]
    im_lat = [im_geotrans[3] + i * im_geotrans[5] for i in range(im_height)]

    im_nc = xr.DataArray(im_data, coords=[im_lat, im_lon], dims=['lat', 'lon'])
    return im_nc


#  1982~1990
filepath='G:/GIMMS_LAI4g/all/GIMMS_LAI4g_AVHRR_MODIS_consolidated_1982_1990/' #文件路径
for y in range(1982, 1991):  # 遍历年
    for m in range(1, 13): 
        for a in range(1,3):
            day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
            filename = 'GIMMS_LAI4g_V1.2_'+str(y) + str(m).zfill(2) +str(a).zfill(2)+'.tif'  # 文件名
    #         print(filename)GIMMS_LAI4g(AVHRR)_V1.2_20110101
            day_nc = tiff2nc(filepath+filename)
            
            time = pd.Timestamp(str(y) + str(m).zfill(2) +str(a).zfill(2))
            day_nc = day_nc.expand_dims(time=[time])
            day_nc = day_nc.where(day_nc != 65535, np.nan)
            day_nc = day_nc/1000
            day_nc = day_nc.astype('float32')
            if filename=='GIMMS_LAI4g_V1.2_19820101.tif':
                merged=day_nc
            if filename!='GIMMS_LAI4g_V1.2_19820101.tif':
                merged=xr.concat([merged, day_nc], dim='time')
                

dataset = merged.to_dataset(name="LAI")
dataset.to_netcdf('G:/GIMMS_LAI4g/merged/'+str('dataset1982_1990')+'.nc') 

#  1991~2000

filepath='G:/GIMMS_LAI4g/all/GIMMS_LAI4g_AVHRR_MODIS_consolidated_1991_2000/' #文件路径
for y in range(1991, 2001):  # 遍历年
    for m in range(1, 13): 
        for a in range(1,3):
            day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
            filename = 'GIMMS_LAI4g_V1.2_'+str(y) + str(m).zfill(2) +str(a).zfill(2)+'.tif'  # 文件名
    #         print(filename)GIMMS_LAI4g(AVHRR)_V1.2_20110101
            day_nc = tiff2nc(filepath+filename)
            
            time = pd.Timestamp(str(y) + str(m).zfill(2) +str(a).zfill(2))
            day_nc = day_nc.expand_dims(time=[time])
            day_nc = day_nc.where(day_nc != 65535, np.nan)
            day_nc = day_nc/1000
            day_nc = day_nc.astype('float32')
            if filename=='GIMMS_LAI4g_V1.2_19910101.tif':
                merged=day_nc
            if filename!='GIMMS_LAI4g_V1.2_19910101.tif':
                merged=xr.concat([merged, day_nc], dim='time')

dataset = merged.to_dataset(name="LAI")
dataset.to_netcdf('G:/GIMMS_LAI4g/merged/'+str('dataset1991_2000')+'.nc') 

#  2001~2010

filepath='G:/GIMMS_LAI4g/all/GIMMS_LAI4g_AVHRR_MODIS_consolidated_2001_2010/' #文件路径
for y in range(2001, 2011):  # 遍历年
    for m in range(1, 13): 
        for a in range(1,3):
            day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
            filename = 'GIMMS_LAI4g_V1.2_'+str(y) + str(m).zfill(2) +str(a).zfill(2)+'.tif'  # 文件名
    #         print(filename)GIMMS_LAI4g(AVHRR)_V1.2_20110101
            day_nc = tiff2nc(filepath+filename)
            
            time = pd.Timestamp(str(y) + str(m).zfill(2) +str(a).zfill(2))
            day_nc = day_nc.expand_dims(time=[time])
            day_nc = day_nc.where(day_nc != 65535, np.nan)
            day_nc = day_nc/1000
            day_nc = day_nc.astype('float32')
            if filename=='GIMMS_LAI4g_V1.2_20010101.tif':
                merged=day_nc
            if filename!='GIMMS_LAI4g_V1.2_20010101.tif':
                merged=xr.concat([merged, day_nc], dim='time')
                
dataset = merged.to_dataset(name="LAI")
dataset.to_netcdf('G:/GIMMS_LAI4g/merged/'+str('dataset2001_2010')+'.nc') 

#   2011~2020

filepath='G:/GIMMS_LAI4g/all/GIMMS_LAI4g_AVHRR_MODIS_consolidated_2011_2020/' #文件路径
for y in range(2011, 2021):  # 遍历年
    for m in range(1, 13): 
        for a in range(1,3):
            day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
            filename = 'GIMMS_LAI4g_V1.2_'+str(y) + str(m).zfill(2) +str(a).zfill(2)+'.tif'  # 文件名
    #         print(filename)GIMMS_LAI4g(AVHRR)_V1.2_20110101
            day_nc = tiff2nc(filepath+filename)
            
            time = pd.Timestamp(str(y) + str(m).zfill(2) +str(a).zfill(2))
            day_nc = day_nc.expand_dims(time=[time])
            day_nc = day_nc.where(day_nc != 65535, np.nan)
            day_nc = day_nc/1000
            day_nc = day_nc.astype('float32')
            if filename=='GIMMS_LAI4g_V1.2_20110101.tif':
                merged=day_nc
            if filename!='GIMMS_LAI4g_V1.2_20110101.tif':
                merged=xr.concat([merged, day_nc], dim='time')
                
dataset = merged.to_dataset(name="LAI")
dataset.to_netcdf('G:/GIMMS_LAI4g/merged/'+str('dataset2011_2020')+'.nc') 
# In[]   Extract and merge the rectangular region covering China

f1 =  xr.open_dataset('G:/GIMMS_LAI4g/merged/dataset1982_1990.nc')

v1 = f1.sel(lon=slice(70, 140), lat=slice(55, 15))

# 1991
f2 =  xr.open_dataset('G:/GIMMS_LAI4g/merged/dataset1991_2000.nc')
v2 = f2.sel(lon=slice(70, 140), lat=slice(55, 15))

# 2001
f3 =  xr.open_dataset('G:/GIMMS_LAI4g/merged/dataset2001_2010.nc')
v3 = f3.sel(lon=slice(70, 140), lat=slice(55, 15))

# 2011
f4 =  xr.open_dataset('G:/GIMMS_LAI4g/merged/dataset2011_2020.nc')
v4 = f4.sel(lon=slice(70, 140), lat=slice(55, 15))

combined = xr.concat([v1, v2], dim='time')
combined = xr.concat([combined, v3], dim='time')
combined = xr.concat([combined, v4], dim='time')
combined.to_netcdf('G:/GIMMS_LAI4g/china05/rec1982-2020.nc') 

# In[]   Generate monthly data for the China region (in a rectangular area)

data = xr.open_dataset('G:/GIMMS_LAI4g/china05/rec1982-2020.nc')
pre=data['LAI']
pre = pre.sortby('lat')
pre = pre.resample(time='1ME').mean()

pre.to_netcdf('G:/CN05.1/month/rec1982-2020_mon.nc')

# In[]   Resample the rectangular China region to 0.25° monthly resolution using nearest neighbor

data = xr.open_dataset('G:/GIMMS_LAI4g/china05/rec1982-2020.nc')
pre=data['LAI']
pre = pre.sortby('lat')
pre = pre.resample(time='1ME').mean()


lon = np.arange(70, 140.25, 0.25)  
lat = np.arange(10, 55.25, 0.25)

coords = {'lon': (['lon'], lon),
          'lat': (['lat'], lat)}

pre_1982 = pre.interp(coords=coords,method='nearest')
dataset = pre_1982.to_dataset(name="LAI")
pre_1982.to_netcdf('G:/CN05.1/month/0.25_nearest_rec_mon.nc')
# In[] Clip the original-resolution LAI to the China region with monthly resolution.

def MaskRegion(ds):
    path = 'D:/daily life/class practice/python&linux/eleven/国家矢量.shp'
    shp = gpd.read_file(path)
    ds2=ds.salem.roi(shape=shp)
    return ds2
ds5=MaskRegion(pre)
ds5.to_netcdf('G:/CN05.1/month/month_onlychina.nc')
# In[] Clip the 0.25° resolution monthly LAI to the China region.

def MaskRegion(ds):
    path = 'D:/daily life/class practice/python&linux/eleven/国家矢量.shp'
    shp = gpd.read_file(path)
    ds2=ds.salem.roi(shape=shp)
    return ds2
ds4=MaskRegion(pre_1982)
ds4.to_netcdf('G:/CN05.1/month/0.25_month_onlychina.nc')

# In[]     Verify
f1 = xr.open_dataset('G:/CN05.1/month/month_onlychina.nc')

f2 = xr.open_dataset('G:/CN05.1/month/0.25_month_onlychina.nc')

f1_timeseries = f1['LAI'].mean(dim=['lat', 'lon'])
f1_timeseries=f1_timeseries.resample(time='1YE').mean()

f2_timeseries = f2['LAI'].mean(dim=['lat', 'lon'])
f2_timeseries=f2_timeseries.resample(time='1YE').mean()

x = np.arange(1982, 2021, 1)
fig1 = plt.figure(figsize=(15, 10))
ax = fig1.add_axes([0.1, 0.1, 0.5, 0.3])
ax.set_ylabel('LAI')
ax.set_xlabel('year')

ax.plot(x, f1_timeseries, label='1/12', color='blue', linewidth=1.2)
ax.plot(x, f2_timeseries, label='1/4', color='red', linewidth=1.2)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=2)