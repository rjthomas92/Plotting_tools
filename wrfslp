import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
import netCDF4
import datetime as dt
import imageio as io
from wrf import (to_np, getvar, interplevel, ALL_TIMES, vertcross, CoordPair, smooth2d, get_basemap, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords)

# Import files
ncfilec = Dataset("../COAWST/wrfout_d01_2018_fineKF2_3")
ncfiles = Dataset("../WRF/wrfout_d01_2013_standalone")

# Get the sea level pressure & other useful variables
for i in range(0,50):
    exec(f'slpc{i} = getvar(ncfilec, "slp", timeidx={i})')
    exec(f'wspdc{i} = getvar(ncfilec, "wspd_wdir", timeidx={i})')
    exec(f'sstc{i} = getvar(ncfilec, "temp", timeidx={i})')
    
for i in range(0,70):
    exec(f'slps{i} = getvar(ncfiles, "slp", timeidx={i})')
    exec(f'wspds{i} = getvar(ncfiles, "wspd_wdir", timeidx={i})')
    exec(f'ssts{i} = getvar(ncfiles, "temp", timeidx={i})')

landmask = getvar(ncfilec, "LANDMASK")

slpvalc = []
slpvals = []
timesc = pd.date_range("2018-07-03-00:00", "2018-07-12-00:00", freq="360min")
for i in range (3,40):
    slpmax = np.min(to_np(locals()["slpc"+str(i)]))
    slpvalc.append(slpmax)
for i in range (4,41):
    slpmax = np.min(to_np(locals()["slps"+str(i)]))
    slpvals.append(slpmax)

fig = plt.figure(figsize=(12,6))

plt.plot(times,sfcpress,'k',label='Best Track')
plt.plot(times,slpvals,label='WRF only')
plt.plot(timesc,slpvalc,color='orange',label='WRF+ROMS')
plt.xticks(np.arange(dt.datetime(2018,7,3), dt.datetime(2018,7,12), dt.timedelta(days=2)).astype(dt.datetime))
plt.ylabel('Surface Pressure [hPa]')
plt.xlabel('Date')
plt.title('Minimum Sea Level Pressure')
plt.legend()

||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
fig = plt.figure(figsize=(12,6))

mar_diff = np.array(sfcpress) - np.array(slpvalc)

plt.plot(times,mar_diff,color='orange',label='Best Track')
plt.plot(times,np.zeros(len(times)),'--k',label='WRF only')
plt.title('MSLP Difference')
plt.ylabel('SLP [hPa]')
plt.xlabel('Date')

|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Finding minimum SLP location - clunky, and needs to be optimised when I have time
#slps = to_np(slpc115)
#arr = to_np(np.min(np.min(slpc115)))
#loc = np.where(slps==arr)
#loc
#lats[99][227].values
