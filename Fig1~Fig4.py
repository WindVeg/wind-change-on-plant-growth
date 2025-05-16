# In[] Fig 1

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pwlf



f=xr.open_dataset('G:/CN05.1/month/CN05.1_Win_1982_2020_yearly_025x025.nc') 
f = f.rio.write_crs("EPSG:4326")

f1=xr.open_dataset('G:/CN05.1/month/month_onlychina.nc')
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
# 添加x=0的线
f2_ax2.axhline(y=0, color='black', linewidth=1)
f2_ax2.text( 0- width/2, slopes[0]*10 + 0.2, 'P < 0.01', ha='center', va='bottom', color='blue', fontsize=7.5)
f2_ax2.text( 1- width/2, slopes[1]*10 + 0.05, 'P < 0.01', ha='center', va='bottom', color='blue', fontsize=7.5)
f2_ax2.text(0 + width/2, slopes1[0]*10 + 0.2, 'P < 0.01', ha='center', va='bottom', color='red', fontsize=7.5)
f2_ax2.text(1 + width/2+0.1, slopes1[1]*10 + 0.05, 'P < 0.01', ha='center', va='bottom', color='red', fontsize=7.5)

# In[]    Fig.2a

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

f=xr.open_dataset('G:/CN05.1/month/CN05.1_Win_1982_2020_yearly_025x025.nc') 
f1=f.win
f1 = f1.resample(time='1YE').mean()
f1= f1.sel(time=slice('1982', '2020'))

lat=f1.lat
lon=f1.lon
# f_timeseries = f['win'].mean(dim=['lat', 'lon'])
years = np.arange(1982, 2021, 1)

# breaks = my_pwlf.fit(2)
# slopes = my_pwlf.calc_slopes()
# rsq = my_pwlf.r_squared()
# se = my_pwlf.standard_errors()
chucun=np.zeros((163, 283))
chucunqian=np.zeros((163,283))
chucunhou=np.zeros((163,283))
for i in range(163):
    for j in range(283):
        series_from_array = pd.Series(f1[:,i,j])
        if np.isnan(series_from_array).any():
            chucun[i,j]=np.nan
            chucunqian[i,j], chucunhou[i,j]=np.nan,np.nan
            # print('1')
            continue
        else:
            my_pwlf = pwlf.PiecewiseLinFit(years, f1[:,i,j])
            a=[]
            for m in range(1986,2017):
                x0 = np.array([1982, m, 2020])
                ssr=my_pwlf.fit_with_breaks(x0)
                a.append(ssr)
            chucun[i,j]=a.index(min(a))+1986
            x0 = np.array([1982, chucun[i,j], 2020])
            ssr=my_pwlf.fit_with_breaks(x0)
            slopes = my_pwlf.calc_slopes()
            chucunqian[i,j]=slopes[0]
            chucunhou[i,j]=slopes[1]
            p = my_pwlf.p_values(method='linear')
            if p[2]>0.05:
                chucun[i,j]=np.nan
                # chucunqian[i,j]=np.nan
                # chucunhou[i,j]=np.nan
                
      
                
data_array = xr.DataArray(chucun, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='TPwind')
ds = xr.Dataset({'TPwind': data_array})   
data_array.to_netcdf('G:/GIMMS_LAI4g/china05/TPwind.nc')                 
                


data_array = xr.DataArray(chucun, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='TPwind')
ds = xr.Dataset({'TPwind': data_array})   
data_array.to_netcdf('G:/GIMMS_LAI4g/china05/TPwind_tested.nc')  


data_array = xr.DataArray(chucunqian, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='TPwind')
ds = xr.Dataset({'TPwindbefore': data_array})   
data_array.to_netcdf('G:/GIMMS_LAI4g/china05/TPwindbefore.nc')  

data_array = xr.DataArray(chucunhou, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='TPwind')
ds = xr.Dataset({'TPwindbefore': data_array})   
data_array.to_netcdf('G:/GIMMS_LAI4g/china05/TPwindafter.nc') 

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
def china_map_feature():
    provinces=cfeature.ShapelyFeature(shpreader.Reader('D:/daily life/class practice/python&linux/eleven/2022省矢量.shp').geometries(),
                                      ccrs.PlateCarree(),edgecolor='k',facecolor='none')
    nine_lines=cfeature.ShapelyFeature(shpreader.Reader('D:/daily life/class practice/python&linux/eleven/九段线.shp').geometries(),
                                      ccrs.PlateCarree(),edgecolor='k',facecolor='none')
    return provinces,nine_lines

fig2 = plt.figure(figsize=(15,15),dpi = 300)
proj = ccrs.PlateCarree() 
leftlon, rightlon, lowerlat, upperlat = (70,140,15,55)

norm = mpl.colors.Normalize(vmin=1988,vmax=2012)
colors = ["#f0e0ff", "#957dad", "#4b0082"]
cmap = LinearSegmentedColormap.from_list("custom_purple", colors, N=256)

f2_ax1 = fig2.add_axes([0.1, 0.1, 0.5, 0.3],projection = proj)
f2_ax1.set_xticks(np.arange(leftlon,rightlon+10,10), crs=ccrs.PlateCarree())
f2_ax1.set_yticks(np.arange(lowerlat,upperlat+10,10), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()  
lat_formatter = cticker.LatitudeFormatter()
f2_ax1.xaxis.set_major_formatter(lon_formatter)
f2_ax1.yaxis.set_major_formatter(lat_formatter)
f2_ax1.tick_params(axis='x', pad=7)
f2_ax1.tick_params(axis='both', labelsize=16)

shps=shpreader.Reader('D:/daily life/class practice/python&linux/eleven/国家矢量.shp')
f2_ax1.add_geometries(shps.geometries(),ccrs.PlateCarree(),
                  linewidth=0.5,edgecolor='k',facecolor='none')
provinces,nine_lines=china_map_feature()
f2_ax1.add_feature(cfeature.OCEAN,color='w',zorder=0)
f2_ax1.add_feature(provinces,linewidth=0.3,zorder=1)
f2_ax1.add_feature(nine_lines,linewidth=0.5,zorder=1)

f2_ax1.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())

f2_ax2 = fig2.add_axes([0.5175, 0.096, 0.08, 0.13],projection = proj)
f2_ax2.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
f2_ax2.add_feature(nine_lines,linewidth=0.5,zorder=2)
f2_ax2.add_geometries(shps.geometries(),ccrs.PlateCarree(),
                  linewidth=0.5,edgecolor='k',facecolor='none')

cf= f2_ax1.contourf(lon,lat,chucun,transform=proj,levels=np.arange(1988,2016,4),norm=norm,
                    cmap=cmap,zorder = 0,extend='both')


cbar_ax = fig2.add_axes([0.15, 0.06, 0.4, 0.015])
cb = fig2.colorbar(cf, cax=cbar_ax,orientation='horizontal',
                   label='TP(year)',extend='both')
cb.ax.tick_params(labelsize=16) 
# In[]   Fig.2b

f1 = xr.open_dataset('G:/GIMMS_LAI4g/china05/TPwindbefore.nc')
lat=f1.lat
lon=f1.lon

fig2 = plt.figure(figsize=(15,15),dpi = 300)
proj = ccrs.PlateCarree() 
leftlon, rightlon, lowerlat, upperlat = (70,140,15,55)

norm = TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)
#绘制地图
f2_ax1 = fig2.add_axes([0.1, 0.1, 0.5, 0.3],projection = proj)
f2_ax1.set_xticks(np.arange(leftlon,rightlon+10,10), crs=ccrs.PlateCarree())
f2_ax1.set_yticks(np.arange(lowerlat,upperlat+10,10), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()  # 设置刻度格式为经纬度格式
lat_formatter = cticker.LatitudeFormatter()
f2_ax1.xaxis.set_major_formatter(lon_formatter)
f2_ax1.yaxis.set_major_formatter(lat_formatter)
f2_ax1.tick_params(axis='x', pad=7)
f2_ax1.tick_params(axis='both', labelsize=16)
# f2_ax1.set_title('(b)',loc='left',fontsize =15)
# f2_ax1.text(0.02, 0.98, '(d)', transform=f2_ax1.transAxes, fontsize=15, va='top', ha='left')

shps=shpreader.Reader('D:/daily life/class practice/python&linux/eleven/国家矢量.shp')
f2_ax1.add_geometries(shps.geometries(),ccrs.PlateCarree(),
                  linewidth=0.5,edgecolor='k',facecolor='none')
provinces,nine_lines=china_map_feature()
f2_ax1.add_feature(cfeature.OCEAN,color='w',zorder=0)
f2_ax1.add_feature(provinces,linewidth=0.3,zorder=1)
f2_ax1.add_feature(nine_lines,linewidth=0.5,zorder=1)
# f2_ax1.coastlines(resolution='110m',color='k',linewidth=0.5,zorder=2)
f2_ax1.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())

f2_ax2 = fig2.add_axes([0.5175, 0.096, 0.08, 0.13],projection = proj)
f2_ax2.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
f2_ax2.add_feature(nine_lines,linewidth=0.5,zorder=2)
f2_ax2.add_geometries(shps.geometries(),ccrs.PlateCarree(),
                  linewidth=0.5,edgecolor='k',facecolor='none')

levels = [-0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1]
cf= f2_ax1.contourf(lon,lat,f1.TPwind,transform=proj, levels=levels,norm=norm,
                    cmap="RdYlBu_r",zorder = 0,extend='both')
# cf1= f2_ax1.contourf(lon1,lat1,chucunqian,levels=[0,0.05,1],
#                     zorder=1,hatches=['...', None],colors="none", transform=ccrs.PlateCarree())  


cbar_ax = fig2.add_axes([0.15, 0.06, 0.4, 0.015])
cb = fig2.colorbar(cf, cax=cbar_ax,orientation='horizontal',
                   extend='both')
cb.set_label(r'wind speed trend before TP (m s$^{-1}$ year$^{-1}$)', fontsize=16)
cb.ax.tick_params(labelsize=16) 

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

data = pd.DataFrame(f1.TPwind.values.flatten(), columns=['TPwind']).dropna()
bins = [-1,-0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1,1]

data['bins'] = pd.cut(data['TPwind'], bins)
frequency = data['bins'].value_counts().sort_index()
frequency=frequency/len(data)


midpoints = [interval.mid for interval in frequency.index]
cmap = plt.get_cmap("RdYlBu_r")

colors = cmap(norm(midpoints))

ax_inset = inset_axes(f2_ax1, width="24%", height="25.6%", loc='lower left', bbox_to_anchor=(0.08, 0.01, 1, 1), bbox_transform=f2_ax1.transAxes) 

ax_inset.bar(frequency.index.astype(str), frequency, color=colors, width=0.8, edgecolor='black')
ax_inset.set_ylabel('Frequency', fontsize=13.1)
ax_inset.get_xaxis().set_visible(False)
ax_inset.tick_params(axis='y', labelsize=13.1) 


# In[]   Fig.2c
f1 = xr.open_dataset('G:/GIMMS_LAI4g/china05/TPwindafter.nc')
from matplotlib.colors import TwoSlopeNorm

fig2 = plt.figure(figsize=(15,15),dpi = 300)
proj = ccrs.PlateCarree() 
leftlon, rightlon, lowerlat, upperlat = (70,140,15,55)

norm = TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)
#绘制地图
f2_ax1 = fig2.add_axes([0.1, 0.1, 0.5, 0.3],projection = proj)
f2_ax1.set_xticks(np.arange(leftlon,rightlon+10,10), crs=ccrs.PlateCarree())
f2_ax1.set_yticks(np.arange(lowerlat,upperlat+10,10), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()  # 设置刻度格式为经纬度格式
lat_formatter = cticker.LatitudeFormatter()
f2_ax1.xaxis.set_major_formatter(lon_formatter)
f2_ax1.yaxis.set_major_formatter(lat_formatter)
f2_ax1.tick_params(axis='x', pad=7)
f2_ax1.tick_params(axis='both', labelsize=16)
# f2_ax1.set_title('(b)',loc='left',fontsize =15)
# f2_ax1.text(0.02, 0.98, '(d)', transform=f2_ax1.transAxes, fontsize=15, va='top', ha='left')

shps=shpreader.Reader('D:/daily life/class practice/python&linux/eleven/国家矢量.shp')
f2_ax1.add_geometries(shps.geometries(),ccrs.PlateCarree(),
                  linewidth=0.5,edgecolor='k',facecolor='none')
provinces,nine_lines=china_map_feature()
f2_ax1.add_feature(cfeature.OCEAN,color='w',zorder=0)
f2_ax1.add_feature(provinces,linewidth=0.3,zorder=1)
f2_ax1.add_feature(nine_lines,linewidth=0.5,zorder=1)
# f2_ax1.coastlines(resolution='110m',color='k',linewidth=0.5,zorder=2)
f2_ax1.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())

f2_ax2 = fig2.add_axes([0.5175, 0.096, 0.08, 0.13],projection = proj)
f2_ax2.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
f2_ax2.add_feature(nine_lines,linewidth=0.5,zorder=2)
f2_ax2.add_geometries(shps.geometries(),ccrs.PlateCarree(),
                  linewidth=0.5,edgecolor='k',facecolor='none')

levels = [-0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1]
cf= f2_ax1.contourf(lon,lat,f1.TPwind,transform=proj, levels=levels,norm=norm,
                    cmap="RdYlBu_r",zorder = 0,extend='both')

cbar_ax = fig2.add_axes([0.15, 0.06, 0.4, 0.015])
cb = fig2.colorbar(cf, cax=cbar_ax,orientation='horizontal',
                   extend='both')
cb.set_label(r'wind speed trend after TP (m s$^{-1}$ year$^{-1}$)', fontsize=16)
cb.ax.tick_params(labelsize=16) 

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

data = pd.DataFrame(f1.TPwind.values.flatten(), columns=['TPwind']).dropna()
bins = [-1,-0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1,1]
# use pd.cut 
data['bins'] = pd.cut(data['TPwind'], bins)
frequency = data['bins'].value_counts().sort_index()
frequency=frequency/len(data)


midpoints = [interval.mid for interval in frequency.index]
cmap = plt.get_cmap("RdYlBu_r")

colors = cmap(norm(midpoints))

ax_inset = inset_axes(f2_ax1, width="24%", height="25.6%", loc='lower left', bbox_to_anchor=(0.08, 0.01, 1, 1), bbox_transform=f2_ax1.transAxes) 

ax_inset.bar(frequency.index.astype(str), frequency, color=colors, width=0.8, edgecolor='black')
ax_inset.set_ylabel('Frequency', fontsize=13.1)
ax_inset.get_xaxis().set_visible(False)
ax_inset.tick_params(axis='y', labelsize=13.1)
# In[]   Fig.2d

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
import maskout

f=xr.open_dataset('G:/CN05.1/month/0.25_nearest_rec_mon.nc')
# f1 = xr.open_dataset('G:/GIMMS_LAI4g/china05/TPwind_tested.nc')
f1 = xr.open_dataset('G:/GIMMS_LAI4g/china05/TPwind.nc')

f = f.sel(lat=slice(15, 55), lon=slice(70, 140))
f=f.LAI
f=f.resample(time='1YE').mean()
f1 = f1.sel(lat=slice(15, 55), lon=slice(70, 140))
f1=f1.TPwind
lat=f.lat
lon=f.lon

years = np.arange(1982, 2021, 1)



f = f.fillna(0)
chucunqian=np.zeros((161,281))
chucunhou=np.zeros((161,281))
for i in range(161):
    for j in range(281):
        series_from_array = pd.Series(f[:,i,j])
        if np.isnan(f1[i, j]):
            # chucun[i,j]=np.nan
            chucunqian[i,j], chucunhou[i,j]=np.nan,np.nan
            # print('1')
            continue
        else:
            my_pwlf = pwlf.PiecewiseLinFit(years, f[:,i,j])
            x0 = np.array([1982, f1[i,j], 2020])
            ssr=my_pwlf.fit_with_breaks(x0)
            slopes = my_pwlf.calc_slopes()
            chucunqian[i,j]=slopes[0]
            chucunhou[i,j]=slopes[1]
            # p = my_pwlf.p_values(method='linear')
            # if p[2]>0.05:
            #     chucun[i,j]=np.nan
                
                
                
def china_map_feature():
    provinces=cfeature.ShapelyFeature(shpreader.Reader('D:/daily life/class practice/python&linux/eleven/2022省矢量.shp').geometries(),
                                      ccrs.PlateCarree(),edgecolor='k',facecolor='none')
    nine_lines=cfeature.ShapelyFeature(shpreader.Reader('D:/daily life/class practice/python&linux/eleven/九段线.shp').geometries(),
                                      ccrs.PlateCarree(),edgecolor='k',facecolor='none')
    return provinces,nine_lines

fig2 = plt.figure(figsize=(15,15),dpi = 300)
proj = ccrs.PlateCarree() 
leftlon, rightlon, lowerlat, upperlat = (70,140,15,55)

norm = TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)

f2_ax1 = fig2.add_axes([0.1, 0.1, 0.5, 0.3],projection = proj)
f2_ax1.set_xticks(np.arange(leftlon,rightlon+10,10), crs=ccrs.PlateCarree())
f2_ax1.set_yticks(np.arange(lowerlat,upperlat+10,10), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()  # 设置刻度格式为经纬度格式
lat_formatter = cticker.LatitudeFormatter()
f2_ax1.xaxis.set_major_formatter(lon_formatter)
f2_ax1.yaxis.set_major_formatter(lat_formatter)
f2_ax1.tick_params(axis='x', pad=7)
f2_ax1.tick_params(axis='both', labelsize=16)

shps=shpreader.Reader('D:/daily life/class practice/python&linux/eleven/国家矢量.shp')
f2_ax1.add_geometries(shps.geometries(),ccrs.PlateCarree(),
                  linewidth=0.5,edgecolor='k',facecolor='none')
provinces,nine_lines=china_map_feature()
f2_ax1.add_feature(cfeature.OCEAN,color='w',zorder=0)
f2_ax1.add_feature(provinces,linewidth=0.3,zorder=1)
f2_ax1.add_feature(nine_lines,linewidth=0.5,zorder=1)
f2_ax1.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())

f2_ax2 = fig2.add_axes([0.5175, 0.096, 0.08, 0.13],projection = proj)
f2_ax2.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
f2_ax2.add_feature(nine_lines,linewidth=0.5,zorder=2)
f2_ax2.add_geometries(shps.geometries(),ccrs.PlateCarree(),
                  linewidth=0.5,edgecolor='k',facecolor='none')
levels = [-0.1, -0.05, -0.02,0, 0.02,0.05, 0.1]
cf= f2_ax1.contourf(lon,lat,chucunqian,transform=proj,levels=levels,norm=norm,
                    cmap="BrBG",zorder = 0,extend='both')
clip=maskout.shp2clip(cf,f2_ax1,
                  "D:/2022.summer wind/毕业论文/白化/country1.shp",'China')

cbar_ax = fig2.add_axes([0.15, 0.06, 0.4, 0.015])
cb = fig2.colorbar(cf, cax=cbar_ax,orientation='horizontal',
                   extend='both')
cb.set_label(r'LAI trend before TP (year$^{-1}$)', fontsize=16)
cb.ax.tick_params(labelsize=16) 

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
data = pd.DataFrame(chucunqian.flatten(), columns=['TPwind']).dropna()
bins = [-1,-0.1, -0.05, -0.02,0, 0.02,0.05, 0.1,1]

data['bins'] = pd.cut(data['TPwind'], bins)
frequency = data['bins'].value_counts().sort_index()
frequency=frequency/len(data)


midpoints = [interval.mid for interval in frequency.index]
cmap = plt.get_cmap("BrBG")

colors = cmap(norm(midpoints))

ax_inset = inset_axes(f2_ax1, width="24%", height="25.6%", loc='lower left', bbox_to_anchor=(0.105, 0.01, 1, 1), bbox_transform=f2_ax1.transAxes) 

ax_inset.bar(frequency.index.astype(str), frequency, color=colors, width=0.8, edgecolor='black')
ax_inset.set_ylabel('Frequency', fontsize=13.1)
ax_inset.get_xaxis().set_visible(False)
ax_inset.tick_params(axis='y', labelsize=13.1)




# In[]   Fig.2e

fig2 = plt.figure(figsize=(15,15),dpi = 300)
proj = ccrs.PlateCarree() 
leftlon, rightlon, lowerlat, upperlat = (70,140,15,55)

norm = TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)

f2_ax1 = fig2.add_axes([0.1, 0.1, 0.5, 0.3],projection = proj)
f2_ax1.set_xticks(np.arange(leftlon,rightlon+10,10), crs=ccrs.PlateCarree())
f2_ax1.set_yticks(np.arange(lowerlat,upperlat+10,10), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()  # 设置刻度格式为经纬度格式
lat_formatter = cticker.LatitudeFormatter()
f2_ax1.xaxis.set_major_formatter(lon_formatter)
f2_ax1.yaxis.set_major_formatter(lat_formatter)
f2_ax1.tick_params(axis='x', pad=7)
f2_ax1.tick_params(axis='both', labelsize=16)

shps=shpreader.Reader('D:/daily life/class practice/python&linux/eleven/国家矢量.shp')
f2_ax1.add_geometries(shps.geometries(),ccrs.PlateCarree(),
                  linewidth=0.5,edgecolor='k',facecolor='none')
provinces,nine_lines=china_map_feature()
f2_ax1.add_feature(cfeature.OCEAN,color='w',zorder=0)
f2_ax1.add_feature(provinces,linewidth=0.3,zorder=1)
f2_ax1.add_feature(nine_lines,linewidth=0.5,zorder=1)

f2_ax1.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())

f2_ax2 = fig2.add_axes([0.5175, 0.096, 0.08, 0.13],projection = proj)
f2_ax2.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
f2_ax2.add_feature(nine_lines,linewidth=0.5,zorder=2)
f2_ax2.add_geometries(shps.geometries(),ccrs.PlateCarree(),
                  linewidth=0.5,edgecolor='k',facecolor='none')

levels = [-0.1, -0.05, -0.02,0, 0.02,0.05, 0.1]
cf= f2_ax1.contourf(lon,lat,chucunhou,transform=proj,levels=levels,norm=norm,
                    cmap="BrBG",zorder = 0,extend='both')
clip=maskout.shp2clip(cf,f2_ax1,
                  "D:/2022.summer wind/毕业论文/白化/country1.shp",'China')

cbar_ax = fig2.add_axes([0.15, 0.06, 0.4, 0.015])
cb = fig2.colorbar(cf, cax=cbar_ax,orientation='horizontal',
                   extend='both')
cb.set_label(r'LAI trend after TP (year$^{-1}$)', fontsize=16)
cb.ax.tick_params(labelsize=16) 

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
data = pd.DataFrame(chucunhou.flatten(), columns=['TPwind']).dropna()
bins = [-1,-0.1, -0.05, -0.02,0, 0.02,0.05, 0.1,1]

data['bins'] = pd.cut(data['TPwind'], bins)
frequency = data['bins'].value_counts().sort_index()
frequency=frequency/len(data)


midpoints = [interval.mid for interval in frequency.index]
cmap = plt.get_cmap("BrBG")

colors = cmap(norm(midpoints))

ax_inset = inset_axes(f2_ax1, width="24%", height="25.6%", loc='lower left', bbox_to_anchor=(0.08, 0.01, 1, 1), bbox_transform=f2_ax1.transAxes) 

ax_inset.bar(frequency.index.astype(str), frequency, color=colors, width=0.8, edgecolor='black')
ax_inset.set_ylabel('Frequency', fontsize=13.1)
ax_inset.get_xaxis().set_visible(False)
ax_inset.tick_params(axis='y', labelsize=13.1)
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
import maskout

def china_map_feature():
    provinces=cfeature.ShapelyFeature(shpreader.Reader('D:/daily life/class practice/python&linux/eleven/2022省矢量.shp').geometries(),
                                      ccrs.PlateCarree(),edgecolor='k',facecolor='none')
    nine_lines=cfeature.ShapelyFeature(shpreader.Reader('D:/daily life/class practice/python&linux/eleven/九段线.shp').geometries(),
                                      ccrs.PlateCarree(),edgecolor='k',facecolor='none')
    return provinces,nine_lines

f=xr.open_dataset('G:/CN05.1/month/CN05.1_Win_1982_2020_yearly_025x025.nc') 
f1=f.win
f1 = f1.resample(time='1YE').mean()
f1= f1.sel(time=slice('1982', '2020'))
f1 = f1.sel(lat=slice(15, 55), lon=slice(70, 140))



f2=xr.open_dataset('G:/CN05.1/month/0.25_nearest_rec_mon.nc')
f2=f2.LAI
f2 = f2.fillna(0.0001)
f2 = f2.resample(time='1YE').mean()
f2= f2.sel(time=slice('1982', '2020'))
f2 = f2.sel(lat=slice(15, 55), lon=slice(70, 140))



lat=f1.lat
lon=f1.lon
# f_timeseries = f['win'].mean(dim=['lat', 'lon'])
years = np.arange(1982, 2021, 1)
f1 = f1.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
f2 = f2.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
# 加载 shp 文件
shp_file = 'D:/风和植被/代码专用/quhua/vegzone_alb54.shp'
gdf = gpd.read_file(shp_file)
gdf['region_group'] = gdf['daima'].str[0]
# unique_region_groups = gdf['region_group'].unique()
unique_region_groups = sorted([x for x in gdf['region_group'].unique() if x is not None])


f1 = f1.rio.write_crs("EPSG:4326")
f2 = f2.rio.write_crs("EPSG:4326")
# 定义一个颜色映射
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_region_groups)))  # 生成足够多的颜色

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
    geometries_VI = gdf_VI.geometry.values  # 提取为几何形状数组

    f1_VI = f1.rio.clip(geometries_VI, gdf.crs, drop=False)
    
    f2_VI = f2.rio.clip(geometries_VI, gdf.crs, drop=False)
    v = np.array(f2_VI).mean((0)) 
    mask = v < 0.01

    expanded_mask = mask[np.newaxis, :, :]
    f1_VI = f1_VI.where(~expanded_mask)
    f_timeseries = f1_VI.mean(dim=['lat', 'lon'])

    years = np.arange(1982, 2021, 1)
    my_pwlf = pwlf.PiecewiseLinFit(years, f_timeseries)
    
    # 进行拟合和预测
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
import maskout

def china_map_feature():
    provinces=cfeature.ShapelyFeature(shpreader.Reader('D:/daily life/class practice/python&linux/eleven/2022省矢量.shp').geometries(),
                                      ccrs.PlateCarree(),edgecolor='k',facecolor='none')
    nine_lines=cfeature.ShapelyFeature(shpreader.Reader('D:/daily life/class practice/python&linux/eleven/九段线.shp').geometries(),
                                      ccrs.PlateCarree(),edgecolor='k',facecolor='none')
    return provinces,nine_lines

f=xr.open_dataset('G:/CN05.1/month/0.25_nearest_rec_mon.nc')
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

shp_file = 'D:/风和植被/代码专用/quhua/vegzone_alb54.shp'
gdf = gpd.read_file(shp_file)
gdf['region_group'] = gdf['daima'].str[0]
# unique_region_groups = gdf['region_group'].unique()
unique_region_groups = sorted([x for x in gdf['region_group'].unique() if x is not None])


f1 = f1.rio.write_crs("EPSG:4326")

colors = plt.cm.viridis(np.linspace(0, 1, len(unique_region_groups)))  # 生成足够多的颜色

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

merged_wind.to_csv('D:/风和植被/代码专用/风速分区数据.txt', sep=' ',header=0, index=0)

merged.to_csv('D:/风和植被/代码专用/LAI分区数据.txt', sep=' ',header=0, index=0)
# In[]  Fig.3a
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
# import pwlf
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
# from matplotlib.colors import TwoSlopeNorm
from shapely.geometry import box


def china_map_feature():
    provinces=cfeature.ShapelyFeature(shpreader.Reader('D:/daily life/class practice/python&linux/eleven/2022省矢量.shp').geometries(),
                                      ccrs.PlateCarree(),edgecolor='k',facecolor='none')
    nine_lines=cfeature.ShapelyFeature(shpreader.Reader('D:/daily life/class practice/python&linux/eleven/九段线.shp').geometries(),
                                      ccrs.PlateCarree(),edgecolor='k',facecolor='none')
    return provinces,nine_lines

color_map = {
    'Tropical Monsoon Forest': '#3B82C4',  
    'Subtropical Evergreen Broadleaf Forest': '#89CFF0',  
    'Warm Temperate Deciduous Broadleaf Forest': '#C5E3F0',  
    'Cold Temperate Coniferous Forest': '#A2D5F5',  
    'Temperate Coniferous and Deciduous Mixed Forest': '#FFCC80',  
    'Temperate Grassland': '#FFE082',  
    'Temperate Desert': '#D7CCC8', 
    'Qinghai-Tibet Plateau Alpine Vegetation': '#B0BEC5'  
}



shp_file = 'D:/风和植被/代码专用/quhua/vegzone_alb54.shp'
gdf = gpd.read_file(shp_file)


gdf = gdf.dropna(subset=['区域'])
gdf['names'] = gdf['daima'].str[0]
replace_dict = {
    '寒温带针叶林区域': 'Cold Temperate Coniferous Forest',
    '温带针叶、落叶阔叶混交林': 'Temperate Coniferous and Deciduous Mixed Forest',
    '暖温带落叶阔叶林区域': 'Warm Temperate Deciduous Broadleaf Forest',
    '亚热带常绿阔叶林区域': 'Subtropical Evergreen Broadleaf Forest',
    '热带季风雨林、雨林区域': 'Tropical Monsoon Forest',
    '温带草原区域': 'Temperate Grassland',
    '温带荒漠区域': 'Temperate Desert',
    '青藏高原高寒植被区域': 'Qinghai-Tibet Plateau Alpine Vegetation'
}


gdf['区域'] = gdf['区域'].replace(replace_dict)

gdf['color'] = gdf['区域'].map(color_map)
gdf = gdf.to_crs("EPSG:4326")

gdf_merged = gdf.dissolve(by='names', as_index=False)


bbox = box(70, 15, 140, 55)  
bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=gdf.crs) 
gdf = gpd.clip(gdf_merged, bbox_gdf)
data_wind=pd.read_csv('D:/风和植被/代码专用/风速分区数据.txt',header=None,sep='\s+',usecols=[0,1,2,3,4,5,6],
                      names=['names','beforeTP','beforeTP_percent','afterTP','afterTP_percent','p_value','p_value1'])

data_LAI=pd.read_csv('D:/风和植被/代码专用/LAI分区数据.txt',header=None,sep='\s+',usecols=[0,1,2,3,4,5,6],
                     names=['names','beforeTP','beforeTP_percent','afterTP','afterTP_percent','p_value','p_value1'])

x_before = data_wind['beforeTP']
x_after = data_wind['afterTP']
y_before = data_LAI['beforeTP']
y_after = data_LAI['afterTP']


data_wind['sensitive_before'] = y_before /x_before 

data_wind['sensitive_after'] = y_after /x_after


fig2 = plt.figure(figsize=(15,15),dpi = 300)
proj = ccrs.PlateCarree() 
ax = fig2.add_axes([0.1, 0.1, 0.5, 0.3],projection = proj)


leftlon, rightlon, lowerlat, upperlat = (70,140,15,55)
ax.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree()) 

ax.set_xticks(np.arange(leftlon,rightlon+10,10), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(lowerlat,upperlat+10,10), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()  # 设置刻度格式为经纬度格式
lat_formatter = cticker.LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.tick_params(axis='x', pad=7)
ax.tick_params(axis='both', labelsize=16)
gdf.plot(color=gdf['color'], ax=ax, edgecolor="black", legend=True)


shps=shpreader.Reader('D:/daily life/class practice/python&linux/eleven/国家矢量.shp')
provinces,nine_lines=china_map_feature()
f2_ax2 = fig2.add_axes([0.485, 0.086, 0.08, 0.13],projection = proj)
f2_ax2.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
f2_ax2.add_feature(nine_lines,linewidth=0.5,zorder=2)
f2_ax2.add_geometries(shps.geometries(),ccrs.PlateCarree(),
                   linewidth=0.5,edgecolor='k',facecolor='none')


region_centroids = {
    'Cold Temperate Coniferous Forest': (122.7, 49.8),
    'Temperate Coniferous and Deciduous Mixed Forest': (130, 45),
    'Warm Temperate Deciduous Broadleaf Forest': (117, 36),
    'Subtropical Evergreen Broadleaf Forest': (110, 28),
    'Tropical Monsoon Forest': (110, 17),
    'Temperate Grassland': (119.8, 44),
    'Temperate Desert': (90, 40),
    'Qinghai-Tibet Plateau Alpine Vegetation': (85, 32)
}


region_data = dict(zip(region_centroids.keys(), data_wind['sensitive_after']))

import matplotlib.patches as patches

for region, (x, y) in region_centroids.items():
    value = region_data[region] * 8  

    bar = patches.Rectangle(
        (x - 0.75, y), 1.5, value, linewidth=1.25, edgecolor='#666666', facecolor='#4CAF50', alpha=1, transform=ccrs.PlateCarree()
    )
    ax.add_patch(bar)

    ax.text(
        x, y + value + 0.01, f"{region_data[region]:.2f}", 
        ha='center', va='bottom', fontsize=16, fontweight='bold', color='black', transform=ccrs.PlateCarree()
    )
    
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=color, lw=4, label=region)
    for region, color in color_map.items()]

ax.legend(
    handles=legend_elements,
    # title="region",
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),  
    ncol=2,  
    frameon=False,  
    fontsize=16
)


# In[]   Fig.3b

# import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
# import pwlf
import cartopy.crs as ccrs
from scipy.stats import linregress
# import cartopy.feature as cfeature
# import cartopy.mpl.ticker as cticker
# import cartopy.io.shapereader as shpreader
# from matplotlib.colors import TwoSlopeNorm


color_map = {
    'Tropical Monsoon Forest': '#3B82C4',  
    'Subtropical Evergreen Broadleaf Forest': '#89CFF0',  
    'Warm Temperate Deciduous Broadleaf Forest': '#C5E3F0',  
    'Cold Temperate Coniferous Forest': '#A2D5F5',  
    'Temperate Coniferous and Deciduous Mixed Forest': '#FFCC80',  
    'Temperate Grassland': '#FFE082',  
    'Temperate Desert': '#D7CCC8',  
    'Qinghai-Tibet Plateau Alpine Vegetation': '#B0BEC5'  
}
data_wind=pd.read_csv('D:/风和植被/代码专用/风速分区数据.txt',header=None,sep='\s+',usecols=[0,1,2,3,4,5,6],
                      names=['names','beforeTP','beforeTP_percent','afterTP','afterTP_percent','p_value','p_value1'])

data_LAI=pd.read_csv('D:/风和植被/代码专用/LAI分区数据.txt',header=None,sep='\s+',usecols=[0,1,2,3,4,5,6],
                     names=['names','beforeTP','beforeTP_percent','afterTP','afterTP_percent','p_value','p_value1'])
order = [
    'Tropical Monsoon Forest',
    'Subtropical Evergreen Broadleaf Forest',
    'Warm Temperate Deciduous Broadleaf Forest',
    'Cold Temperate Coniferous Forest',
    'Temperate Coniferous and Deciduous Mixed Forest',
    'Temperate Grassland',
    'Temperate Desert',
    'Qinghai-Tibet Plateau Alpine Vegetation'
]

data_wind['names'] = pd.Categorical(data_wind['names'], categories=order, ordered=True)
data_wind = data_wind.sort_values('names').reset_index(drop=True)


data_LAI['names'] = pd.Categorical(data_LAI['names'], categories=order, ordered=True)
data_LAI = data_LAI.sort_values('names').reset_index(drop=True)

x_before = data_wind['beforeTP']
x_after = data_wind['afterTP']
y_before = data_LAI['beforeTP']
y_after = data_LAI['afterTP']
p_value = data_LAI['p_value']
p_value1 = data_LAI['p_value1']
names = data_LAI['names']



fig, ax = plt.subplots(figsize=(8, 6))

for name, xb, xa, yb, ya, p,p1 in zip(names, x_before, x_after, y_before, y_after, p_value,p_value1):
    color = color_map[name]  

    ax.scatter(xb, yb, color=color, alpha=0.9, label=name if name not in [label.get_text() for label in ax.legend().get_texts()] else "", marker='o', s=250,
               edgecolors='black' if p < 0.01 else None, linewidths=1.5 if p < 0.01 else 0.5)

    ax.scatter(xa, ya, color=color, alpha=0.9, label=name if name not in [label.get_text() for label in ax.legend().get_texts()] else "", marker='^', s=250,
               edgecolors='black' if p < 0.01 else None, linewidths=1.5 if p < 0.01 else 0.5)




x_combined = np.concatenate([x_before, x_after])  
y_combined = np.concatenate([y_before, y_after])  

slope, intercept, r_value, p_value_reg, std_err = linregress(x_combined, y_combined)


x_fit = np.linspace(min(x_combined), max(x_combined), 500)
y_fit = slope * x_fit + intercept


ax.plot(x_fit, y_fit, color='black', linestyle='-', linewidth=1.5)


r_squared = r_value**2

p_text = "$p < 0.01$" if p_value_reg < 0.01 else f"$p = {p_value_reg:.3f}$"

r_squared = r_value**2
ax.text(0.05, 0.95, f"$R^2 = {r_squared:.2f}$\n{p_text}",
        transform=ax.transAxes, fontsize=16, verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

ax.tick_params(axis='both', labelsize=16)

ax.set_xlabel('wind speed change rate(m s$^{-1}$ year$^{-1}$)', fontsize=16)
ax.set_ylabel('LAI change rate(year$^{-1}$)', fontsize=16)
ax.axhline(y=0, color='black', linewidth=1, linestyle='--') 
ax.axvline(x=0, color='black', linewidth=1, linestyle='--')  

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=2)

# In[]  Fig.4 Data generation process

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
import statsmodels.api as sm
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import maskout
def china_map_feature():
    provinces=cfeature.ShapelyFeature(shpreader.Reader('D:/daily life/class practice/python&linux/eleven/2022省矢量.shp').geometries(),
                                      ccrs.PlateCarree(),edgecolor='k',facecolor='none')
    nine_lines=cfeature.ShapelyFeature(shpreader.Reader('D:/daily life/class practice/python&linux/eleven/九段线.shp').geometries(),
                                      ccrs.PlateCarree(),edgecolor='k',facecolor='none')
    return provinces,nine_lines


f1=xr.open_dataset('G:/CN05.1/annual/CN05.1_Tm_1961_2022_annual_025x025.nc')
f1 = f1.sel(lat=slice(15, 55), lon=slice(70, 140))
f1= f1.sel(time=slice('1982', '2020'))

f2=xr.open_dataset('G:/CN05.1/annual/CN05.1_Pre_1961_2022_annual_025x025.nc') 
f2 = f2.sel(lat=slice(15, 55), lon=slice(70, 140))
f2= f2.sel(time=slice('1982', '2020'))

f3=xr.open_dataset('G:/CN05.1/annual/CN05.1_Win_1961_2022_annual_025x025.nc') 
f3 = f3.sel(lat=slice(15, 55), lon=slice(70, 140))
f3= f3.sel(time=slice('1982', '2020'))

f = xr.open_dataset('G:/CN05.1/month/0.25_month_onlychina.nc')
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


# 激活pandas与R的数据框转换
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
data_array.to_netcdf('G:/GIMMS_LAI4g/china05/tm.nc') 

data_array = xr.DataArray(chucunpre*100, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='tm')
data_array.to_netcdf('G:/GIMMS_LAI4g/china05/pre.nc') 

data_array = xr.DataArray(chucunwin*100, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='tm')
data_array.to_netcdf('G:/GIMMS_LAI4g/china05/win.nc') 

data_array = xr.DataArray(chucunr2*100, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon}, name='tm')
data_array.to_netcdf('G:/GIMMS_LAI4g/china05/r2.nc') 
# In[]  fig4 
from matplotlib.colors import ListedColormap

tm=xr.open_dataset('G:/GIMMS_LAI4g/china05/tm.nc')
pre=xr.open_dataset('G:/GIMMS_LAI4g/china05/pre.nc')
win=xr.open_dataset('G:/GIMMS_LAI4g/china05/win.nc')

lon=tm.lon
lat=tm.lat


result = np.full((161, 281), np.nan)


for i in range(161):
   for j in range(281):
       values = [
           ('tm', tm.tm.values[i,j]),
           ('pre', pre.tm.values[i,j]), 
           ('win', win.tm.values[i,j])
       ]

       valid_values = [(name, val) for name, val in values if not np.isnan(val)]
       if valid_values:
    
           max_name = max(valid_values, key=lambda x: x[1])[0]
         
           label_map = {'tm':0, 'pre':1, 'win':2}
           result[i,j] = label_map[max_name]
           
           
           
           
def china_map_feature():
    provinces=cfeature.ShapelyFeature(shpreader.Reader('D:/daily life/class practice/python&linux/eleven/2022省矢量.shp').geometries(),
                                      ccrs.PlateCarree(),edgecolor='k',facecolor='none')
    nine_lines=cfeature.ShapelyFeature(shpreader.Reader('D:/daily life/class practice/python&linux/eleven/九段线.shp').geometries(),
                                      ccrs.PlateCarree(),edgecolor='k',facecolor='none')
    return provinces,nine_lines

fig2 = plt.figure(figsize=(15,15),dpi = 300)
proj = ccrs.PlateCarree() 
f2_ax1 = fig2.add_axes([0.1, 0.1, 0.5, 0.3],projection = proj)
leftlon, rightlon, lowerlat, upperlat = (70,140,15,55)
colors = ['pink', 'skyblue', 'yellow'] 
cmap = ListedColormap(colors)
norm = plt.Normalize(vmin=0, vmax=3)

f2_ax1 = fig2.add_axes([0.1, 0.1, 0.5, 0.3],projection = proj)
f2_ax1.set_xticks(np.arange(leftlon,rightlon+10,10), crs=ccrs.PlateCarree())
f2_ax1.set_yticks(np.arange(lowerlat,upperlat+10,10), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()  
lat_formatter = cticker.LatitudeFormatter()
f2_ax1.xaxis.set_major_formatter(lon_formatter)
f2_ax1.yaxis.set_major_formatter(lat_formatter)
f2_ax1.tick_params(axis='x', pad=7)
f2_ax1.tick_params(axis='both', labelsize=16)
shps=shpreader.Reader('D:/daily life/class practice/python&linux/eleven/国家矢量.shp')
f2_ax1.add_geometries(shps.geometries(),ccrs.PlateCarree(),
                  linewidth=0.5,edgecolor='k',facecolor='none')
provinces,nine_lines=china_map_feature()
f2_ax1.add_feature(cfeature.OCEAN,color='w',zorder=0)
f2_ax1.add_feature(provinces,linewidth=0.3,zorder=1)
f2_ax1.add_feature(nine_lines,linewidth=0.5,zorder=1)

f2_ax1.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())

f2_ax2 = fig2.add_axes([0.5175, 0.096, 0.08, 0.13],projection = proj)
f2_ax2.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
f2_ax2.add_feature(nine_lines,linewidth=0.5,zorder=2)
f2_ax2.add_geometries(shps.geometries(),ccrs.PlateCarree(),
                  linewidth=0.5,edgecolor='k',facecolor='none')


cf= f2_ax1.contourf(lon,lat,result,transform=proj,
                    cmap=cmap)

cbar_ax = fig2.add_axes([0.15, 0.06, 0.4, 0.015])


cb = fig2.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax, orientation='horizontal', 
                  ticks=np.arange(1, 4) - 0.5)
cb.set_ticklabels([ 'tm', 'pre', 'win'])
cb.ax.tick_params(labelsize=16) 
