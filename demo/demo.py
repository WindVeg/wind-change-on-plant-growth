# In[] Fig 1  piecewise linear regression

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pwlf



f=xr.open_dataset('wind-change-on-plant-growth/wind data/CN05.1_Win_1982_2020_yearly_025x025.nc') 
f = f.rio.write_crs("EPSG:4326")

f1=xr.open_dataset('wind-change-on-plant-growth/LAI data process/0.25_month_onlychina.nc')
f1 = f1.rio.write_crs("EPSG:4326")
lon = np.arange(69.75, 140.5, 0.25)  
lat = np.arange(14.75, 55.5, 0.25)

coords = {'lon': (['lon'], lon),
          'lat': (['lat'], lat)}

f1 = f1.interp(coords=coords,method='nearest')
f1=f1.resample(time='1YE').mean()
f1 = f1.fillna(0)




v = np.array(f1['LAI']).mean((0)) 
mask = v < 0.01
expanded_mask = mask[np.newaxis, :, :]

f1['LAI'] = f1['LAI'].where(~expanded_mask)
f['win'] = f['win'].where(~expanded_mask)

f_timeseries = f['win'].mean(dim=['lat', 'lon'])
years = np.arange(1982, 2021, 1)
my_pwlf = pwlf.PiecewiseLinFit(years, f_timeseries)
a=[]
for m in range(1986,2017):
    x0 = np.array([1982, m, 2020])
    ssr=my_pwlf.fit_with_breaks(x0)
    a.append(ssr)
a=a.index(min(a))+1986
ssr=my_pwlf.fit_with_breaks(np.array([1982, a, 2020]))
# breaks = my_pwlf.fit(2)
slopes = my_pwlf.calc_slopes()
rsq = my_pwlf.r_squared()
# se = my_pwlf.standard_errors()
xHat = np.linspace(min(years), max(years), num=10000)
yHat = my_pwlf.predict(xHat)
p = my_pwlf.p_values(method='linear')
beta = my_pwlf.beta
# t = my_pwlf.beta / my_pwlf.se


f1_timeseries = f1['LAI'].mean(dim=['lat', 'lon'])
years1 = np.arange(1982, 2021, 1)
my_pwlf1 = pwlf.PiecewiseLinFit(years1, f1_timeseries)
a1=[]
# for m in range(1986,2017):
#     x0 = np.array([1982, m, 2020])
#     ssr=my_pwlf1.fit_with_breaks(x0)
#     a1.append(ssr)
# a1=a1.index(min(a1))+1986
ssr=my_pwlf1.fit_with_breaks(np.array([1982, a, 2020]))
p1 = my_pwlf1.p_values(method='linear')
slopes1 = my_pwlf1.calc_slopes()
beta1 = my_pwlf1.beta
rsq1 = my_pwlf1.r_squared()

xHat1 = np.linspace(min(years1), max(years1), num=10000)
yHat1 = my_pwlf1.predict(xHat1)


fig1 = plt.figure(figsize=(15, 12))
ax = fig1.add_axes([0.1, 0.1, 0.5, 0.3])
ax.set_ylabel('wind speed')
ax.set_ylim(2.1, 3)
ax.set_yticks([2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3])
ax.plot(years, f_timeseries.values, label='wind', linestyle='--',color='blue', linewidth=2)
ax.plot(xHat, yHat, color='blue', linewidth=2)

ax2 = ax.twinx()
ax2.set_ylabel('LAI')
ax2.set_ylim(1.1, 1.5)
ax2.set_yticks([1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5])
ax2.plot(years1, f1_timeseries,linestyle='--', label='LAI', color='red', linewidth=2)
ax2.plot(xHat1, yHat1,  color='red', linewidth=2)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True,ncol=2)

ax.text(0.65, 0.81, "Turning point=2010", 
        fontsize=12, color='black', ha='center', va='center', transform=ax.transAxes)
ax.text(0.65, 0.74, "R²=0.73, P<0.001", 
        fontsize=12, color='blue', ha='center', va='center', transform=ax.transAxes)
ax.text(0.65, 0.68, "R²=0.89, P<0.001", 
        fontsize=12, color='red', ha='center', va='center', transform=ax.transAxes)


ax.axvline(x=2010, color='black', linewidth=1, linestyle='--', alpha=0.5)

f2_ax2 = fig1.add_axes([0.161, 0.12, 0.12, 0.05])


x = np.arange(2)
width = 0.35

f2_ax2.bar(x - width/2, slopes*10, width, label='Wind Trend', color='blue')

f2_ax2.bar(x + width/2, slopes1*10, width, label='LAI Trend', color='red')
f2_ax2.set_xticks(x)
f2_ax2.set_xticklabels(['Before TP', 'After TP'])
f2_ax2.spines['top'].set_visible(False)
f2_ax2.spines['right'].set_visible(False)
f2_ax2.tick_params(axis='y', labelsize=8)

f2_ax2.axhline(y=0, color='black', linewidth=1)
f2_ax2.text( 0- width/2, slopes[0]*10 + 0.2, 'P < 0.01', ha='center', va='bottom', color='blue', fontsize=7.5)
f2_ax2.text( 1- width/2, slopes[1]*10 + 0.05, 'P < 0.01', ha='center', va='bottom', color='blue', fontsize=7.5)
f2_ax2.text(0 + width/2, slopes1[0]*10 + 0.2, 'P < 0.01', ha='center', va='bottom', color='red', fontsize=7.5)
f2_ax2.text(1 + width/2+0.1, slopes1[1]*10 + 0.05, 'P < 0.01', ha='center', va='bottom', color='red', fontsize=7.5)


# In[] generate wind speed turning point of different vegetation zone 
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pwlf
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
from matplotlib.colors import TwoSlopeNorm



f=xr.open_dataset('wind-change-on-plant-growth/wind data/CN05.1_Win_1982_2020_yearly_025x025.nc') 
f1=f.win
f1 = f1.resample(time='1YE').mean()
f1= f1.sel(time=slice('1982', '2020'))
f1 = f1.sel(lat=slice(15, 55), lon=slice(70, 140))



f2=xr.open_dataset('wind-change-on-plant-growth/demo/0.25_nearest_rec_mon_year.nc')
f2=f2.LAI
f2 = f2.fillna(0.0001)
f2 = f2.resample(time='1YE').mean()
f2= f2.sel(time=slice('1982', '2020'))
f2 = f2.sel(lat=slice(15, 55), lon=slice(70, 140))



lat=f1.lat
lon=f1.lon

years = np.arange(1982, 2021, 1)
f1 = f1.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
f2 = f2.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

shp_file = 'wind-change-on-plant-growth/demo/quhua/vegzone_alb54.shp'  
gdf = gpd.read_file(shp_file)
gdf['region_group'] = gdf['daima'].str[0]
# unique_region_groups = gdf['region_group'].unique()
unique_region_groups = sorted([x for x in gdf['region_group'].unique() if x is not None])



colors = plt.cm.viridis(np.linspace(0, 1, len(unique_region_groups)))  

fig1 = plt.figure(figsize=(15, 10))
ax = fig1.add_axes([0.1, 0.1, 0.5, 0.3])
ax.set_ylabel('wind')

names = [
    'Cold Temperate Coniferous Forest',
    'Temperate Coniferous and Deciduous Mixed Forest',
    'Warm Temperate Deciduous Broadleaf Forest',
    'Subtropical Evergreen Broadleaf Forest',
    'Tropical Monsoon Forest',
    'Temperate Grassland',
    'Temperate Desert',
    'Qinghai-Tibet Plateau Alpine Vegetation'
]
TP=[]
p_value=[]
p_value1=[]
chucunqian=[]
chucunqian_percent=[]
chucunhou=[]
chucunhou_percent=[]

for idx, i in enumerate(unique_region_groups):
    gdf_VI = gdf[gdf['region_group'] == i]
    geometries_VI = gdf_VI.geometry.values  

    f1_VI = f1.rio.clip(geometries_VI, gdf.crs, drop=False)
    
    f2_VI = f2.rio.clip(geometries_VI, gdf.crs, drop=False)
    v = np.array(f2_VI).mean((0)) 
    mask = v < 0.01

    expanded_mask = mask[np.newaxis, :, :]
    f1_VI = f1_VI.where(~expanded_mask)
    f_timeseries = f1_VI.mean(dim=['lat', 'lon'])

    years = np.arange(1982, 2021, 1)
    my_pwlf = pwlf.PiecewiseLinFit(years, f_timeseries)
    

    a = []
    for m in range(1986, 2017):
        x0 = np.array([1982, m, 2020])
        ssr = my_pwlf.fit_with_breaks(x0)
        a.append(ssr)
    a = a.index(min(a)) + 1986
    TP.append(a)
    ssr = my_pwlf.fit_with_breaks(np.array([1982, a, 2020]))
    p = my_pwlf.p_values(method='linear')
    p_value1.append(p[1])
    p_value.append(p[2])
    slopes = my_pwlf.calc_slopes()
    chucunqian.append(slopes[0])
    chucunhou.append(slopes[1])
    chucunqian_percent.append(100*slopes[0]/f_timeseries.values.mean())
    chucunhou_percent.append(100*slopes[1]/f_timeseries.values.mean())
    
    xHat = np.linspace(min(years), max(years), num=10000)
    yHat = my_pwlf.predict(xHat)

    ax.plot(years, f_timeseries.values, label=names[idx], linestyle='--', color=colors[idx], linewidth=2)

    ax.plot(xHat, yHat, color=colors[idx], linewidth=2)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=3)

plt.title('wind Time Series by Region Group')


merged_wind = pd.DataFrame({
    "names": names,
    # 'TPwind': TP,
    'beforeTP': chucunqian,
    "beforeTP_percent": chucunqian_percent,
    'afterTP': chucunhou,
    "afterTP_percent": chucunhou_percent,
    'P_value': p_value,
    'P_value1': p_value1
})

# In[]  Analyze the trend changes of LAI over different vegetation zones and time periods,
 #      using TP years from wind speed data, run previous cell first
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pwlf
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
from matplotlib.colors import TwoSlopeNorm



f=xr.open_dataset('wind-change-on-plant-growth/demo/0.25_nearest_rec_mon_year.nc')
f1=f.LAI
f1 = f1.fillna(0.0001)

f1 = f1.resample(time='1YE').mean()
f1= f1.sel(time=slice('1982', '2020'))
f1 = f1.sel(lat=slice(15, 55), lon=slice(70, 140))


lat=f1.lat
lon=f1.lon
# f_timeseries = f['win'].mean(dim=['lat', 'lon'])
years = np.arange(1982, 2021, 1)
f1 = f1.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

shp_file = 'wind-change-on-plant-growth/demo/quhua/vegzone_alb54.shp'
gdf = gpd.read_file(shp_file)
gdf['region_group'] = gdf['daima'].str[0]
# unique_region_groups = gdf['region_group'].unique()
unique_region_groups = sorted([x for x in gdf['region_group'].unique() if x is not None])


f1 = f1.rio.write_crs("EPSG:4326")

colors = plt.cm.viridis(np.linspace(0, 1, len(unique_region_groups)))  

fig1 = plt.figure(figsize=(15, 10))
ax = fig1.add_axes([0.1, 0.1, 0.5, 0.3])
ax.set_ylabel('LAI')

names = [
    'Cold Temperate Coniferous Forest',
    'Temperate Coniferous and Deciduous Mixed Forest',
    'Warm Temperate Deciduous Broadleaf Forest',
    'Subtropical Evergreen Broadleaf Forest',
    'Tropical Monsoon Forest',
    'Temperate Grassland',
    'Temperate Desert',
    'Qinghai-Tibet Plateau Alpine Vegetation'
]
p_value=[]
p_value1=[]
chucunqian=[]
chucunqian_percent=[]
chucunhou=[]
chucunhou_percent=[]

for idx, i in enumerate(unique_region_groups):
    gdf_VI = gdf[gdf['region_group'] == i]
    geometries_VI = gdf_VI.geometry.values  

    f1_VI = f1.rio.clip(geometries_VI, gdf.crs, drop=False)
    v = np.array(f1_VI).mean((0)) 
    mask = v < 0.01

    expanded_mask = mask[np.newaxis, :, :]
    f1_VI = f1_VI.where(~expanded_mask)
    f_timeseries = f1_VI.mean(dim=['lat', 'lon'])

    years = np.arange(1982, 2021, 1)
    my_pwlf = pwlf.PiecewiseLinFit(years, f_timeseries)
    
    # 进行拟合和预测
    ssr = my_pwlf.fit_with_breaks(np.array([1982, TP[idx], 2020]))
    p = my_pwlf.p_values(method='linear')
    p_value1.append(p[1])
    p_value.append(p[2])
    slopes = my_pwlf.calc_slopes()
    chucunqian.append(slopes[0])
    chucunhou.append(slopes[1])
    chucunqian_percent.append(100*slopes[0]/f_timeseries.values.mean())
    chucunhou_percent.append(100*slopes[1]/f_timeseries.values.mean())

    xHat = np.linspace(min(years), max(years), num=10000)
    yHat = my_pwlf.predict(xHat)


    ax.plot(years, f_timeseries.values, label=names[idx], linestyle='--', color=colors[idx], linewidth=2)

    ax.plot(xHat, yHat, color=colors[idx], linewidth=2)

# 添加图例
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=3)

plt.title('LAI Time Series by Region Group')

merged = pd.DataFrame({
    "names": names,
    # 'TP': TP,
    'beforeTP': chucunqian,
    "beforeTP_percent": chucunqian_percent,
    'afterTP': chucunhou,
    "afterTP_percent": chucunhou_percent,
    'P_value': p_value,
    'P_value1': p_value1
})

merged_wind.to_csv('path/windzone.txt', sep=' ',header=0, index=0)

merged.to_csv('path/LAIzone.txt', sep=' ',header=0, index=0)


# In[]  Fig.4 Data generation process
import os
os.environ['R_HOME']=r'replace by your_R_home_path'  #'replace by your_R_home_path' can get from R, using the following order:R.home()
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import pandas as pd
from rpy2.robjects import pandas2ri
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
# import pwlf
import cartopy.crs as ccrs
# from scipy.stats import linregress
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
# from matplotlib.colors import TwoSlopeNorm

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


f1=xr.open_dataset('wind-change-on-plant-growth/demo/CN05.1_Tm_1961_2022_annual_025x025.nc')
f1 = f1.sel(lat=slice(15, 55), lon=slice(70, 140))
f1= f1.sel(time=slice('1982', '2020'))

f2=xr.open_dataset('wind-change-on-plant-growth/demo/CN05.1_Pre_1961_2022_annual_025x025.nc') 
f2 = f2.sel(lat=slice(15, 55), lon=slice(70, 140))
f2= f2.sel(time=slice('1982', '2020'))

f3=xr.open_dataset('wind-change-on-plant-growth/demo/CN05.1_Win_1961_2022_annual_025x025.nc') 
f3 = f3.sel(lat=slice(15, 55), lon=slice(70, 140))
f3= f3.sel(time=slice('1982', '2020'))

f = xr.open_dataset('wind-change-on-plant-growth/LAI data process/0.25_month_onlychina.nc')
f=f.resample(time='1YE').mean()
f = f.sel(lat=slice(15, 55), lon=slice(70, 140))
v = np.array(f['LAI']).mean((0)) 
v = np.nan_to_num(v, nan=0)
mask = v < 0.01
expanded_mask = mask[np.newaxis, :, :]
f['LAI'] = f['LAI'].where(~expanded_mask)

f3['win'] = f3['win'].where(~expanded_mask)
f1['tm'] = f1['tm'].where(~expanded_mask)
f2['pre'] = f2['pre'].where(~expanded_mask)

lon=f1.lon
lat=f1.lat



pandas2ri.activate()
relaimpo = importr('relaimpo')
stats = importr('stats')
base = importr('base')

chucuntm=np.zeros((161,281))
chucunpre=np.zeros((161,281))
chucunwin=np.zeros((161,281))
chucunr2=np.zeros((161,281))
chucunr21=np.zeros((161,281))
chucunr2cha=np.zeros((161,281))
for i in range(161):
    for j in range(281):
        if np.isnan(f3['win'][:,i,j]).any():
            # chucun[i,j]=np.nan
            chucuntm[i,j], chucunpre[i,j]=np.nan,np.nan
            chucunwin[i,j], chucunr2[i,j]=np.nan,np.nan
            # print('1')
            continue
        else:

            data = pd.DataFrame({
                'y': f['LAI'][:,i,j].values,
                'x1': f1['tm'][:,i,j].values,
                'x2': f2['pre'][:,i,j].values,
                'x3': f3['win'][:,i,j].values
            })
            r_dataframe = pandas2ri.py2rpy(data)
            data1 = pd.DataFrame({
                'y': f['LAI'][:,i,j].values,
                'x1': f1['tm'][:,i,j].values,
                'x2': f2['pre'][:,i,j].values,
            })
            r_dataframe1 = pandas2ri.py2rpy(data1)
            
            lm_model = stats.lm('y ~ x1 + x2 + x3', data=r_dataframe)
            importance = relaimpo.calc_relimp(lm_model, type='lmg', rela=True)
            lmg_values = np.array(importance.do_slot('lmg'))
            # r2 = importance.do_slot('R2')
            chucuntm[i,j]=lmg_values[0]
            chucunpre[i,j]=lmg_values[1]
            chucunwin[i,j]=lmg_values[2]
            chucunr2[i,j]=importance.do_slot('R2')
         

data_array = xr.DataArray(chucuntm*100, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='tm')
data_array.to_netcdf('Your path.nc') 

data_array = xr.DataArray(chucunpre*100, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='tm')
data_array.to_netcdf('Your path.nc') 

data_array = xr.DataArray(chucunwin*100, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='tm')
data_array.to_netcdf('Your path.nc') 

data_array = xr.DataArray(chucunr2*100, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='tm')
data_array.to_netcdf('Your path.nc') 
