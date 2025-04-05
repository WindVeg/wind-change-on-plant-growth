
import numpy as np
import pandas as pd
import xarray as xr


f=xr.open_dataset('G:/CN05.1/CN05.1_Win_1961_2022_daily_025x025.nc')

pre=f.win
pre = pre.resample(time='1ME').mean()
pre.to_netcdf('G:/CN05.1/month/CN05.1_Win_1961_2022_monthly_025x025.nc') 


f1=xr.open_dataset('G:/CN05.1/month/CN05.1_Win_1961_2022_monthly_025x025.nc')