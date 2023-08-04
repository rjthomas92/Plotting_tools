import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from eofs.standard import Eof
import cartopy.crs as ccrs
import wget

proj               = ccrs.PlateCarree(central_longitude=180)
trans              = ccrs.PlateCarree()
ersst              = xr.open_mfdataset('Data/ERSST/Summer/ersst.v5.*.nc',parallel=True, concat_dim="time",
                        combine="nested", data_vars='minimal', coords='minimal', compat='override')
sst_obs_xa         = ersst['sst'].sel(lat=slice(0,80))
sst_obs_na         = sst_obs_xa.values
tsz,e,ysz,xsz      = sst_obs_xa.shape
yrsz               = int(tsz/3)
sst_obs_ann        = sst_obs_na.reshape(yrsz,3,ysz,xsz).mean(axis=1)
time               = str(ersst['time'][0].values)
yrs                = np.arange(yrsz)+pd.to_datetime(time).year
lon, lat           = sst_obs_xa['lon'].values, sst_obs_xa['lat'].values

weights            = np.sqrt(np.cos((lat.reshape(ysz,1)@np.ones([1,xsz]))/180*np.pi))
solver             = Eof(sst_obs_ann,weights=weights)
eofs_reg           = solver.eofsAsCovariance(neofs=2)
eofs_cor           = solver.eofsAsCorrelation(neofs=2)
variance_fractions = solver.varianceFraction(neigs=3)
VARI               = variance_fractions
pcs         = solver.pcs(npcs=3,pcscaling=1)

pnl_pos            = [[0.05, 0.50, 0.40, 0.45],[0.55, 0.50, 0.40, 0.45]]
cbar_pos           = [0.10, 0.48, 0.80, 0.03]
pnl_tsrs           = [0.10, 0.05, 0.80, 0.35]
msz_fig            = 2

fig       = plt.figure(figsize=(15,15))
f2_ax1    = fig.add_axes([0.1, 0.55, 0.3, 0.2])
c2        = f2_ax1.contourf(lon,lat ,eofs_cor[0,:,:], levels=np.arange(-1,1.1,0.1), extend = 'both',zorder=0,cmap=plt.cm.RdBu_r,facecolor='grey')
f2_ax1.contour(lon,lat ,eofs_reg[0,:,:], levels=np.arange(-1,1.1,0.1), extend = 'both',zorder=0,cmap=plt.cm.RdBu_r,facecolor='grey')

f2_ax1.set_facecolor('lightgrey')
f2_ax1.set_ylabel('Latitude',fontsize=15)
f2_ax1.set_xlabel('Longitude',fontsize=15)
f2_ax1.set_yticks([0,30,60])
f2_ax1.set_yticklabels([r'0$^\degree$',r'30$^\degree$N',r'60$^\degree$N'])
f2_ax1.set_xticks([0,60,120,180,240,300])
f2_ax1.set_xticklabels([r'0$^\degree$',r'60$^\degree$E', r'120$^\degree$E',r'180$^\degree$',r'60$^\degree$W',r'120$^\degree$W'])
f2_ax1.set_title('(a) EOF1',loc='left')
f2_ax1.set_title( '%.2f%%' % (variance_fractions[0]*100),loc='right')
f2_ax1    = fig.add_axes([0.1, 0.25, 0.3, 0.2])
f2_ax1.set_facecolor('lightgrey')
f2_ax1.set_title('(c) EOF2',loc='left')
f2_ax1.set_title( '%.2f%%' % (variance_fractions[1]*100),loc='right')
f2_ax1.set_ylabel('Latitude',fontsize=15)
f2_ax1.set_xlabel('Longitude',fontsize=15)
f2_ax1.set_yticks([0,30,60])
f2_ax1.set_yticklabels([r'0$^\degree$',r'30$^\degree$N',r'60$^\degree$N'])
f2_ax1.set_xticks([0,60,120,180,240,300])
f2_ax1.set_xticklabels([r'0$^\degree$',r'60$^\degree$E', r'120$^\degree$E',r'180$^\degree$',r'60$^\degree$W',r'120$^\degree$W'])
f2_ax1.contourf(lon,lat,eofs_cor[1,:,:], levels=np.arange(-1,1.1,0.1), extend = 'both',zorder=0, cmap=plt.cm.RdBu_r,facecolor='grey')
f2_ax1.contour(lon,lat,eofs_reg[1,:,:], levels=np.arange(-1,1.1,0.1), extend = 'both',zorder=0, cmap=plt.cm.RdBu_r,facecolor='grey')

f2_ax2    = fig.add_axes([0.45, 0.55, 0.3, 0.2])
f2_ax2.set_title('(b) PC1',loc='left')
plt.ylim(-3,3)
f2_ax2.axhline(0,linestyle="--")
f2_ax2.plot(yrs,pcs[:,0],c='k')
f2_ax3    = fig.add_axes([0.45, 0.25, 0.3, 0.2])
f2_ax3.set_title('(d) PC2',loc='left')
plt.ylim(-3,3)
f2_ax3.axhline(0,linestyle="--")
f2_ax3.plot(yrs,pcs[:,1],c='k')
position  = fig.add_axes([0.1, 0.15, 0.3, 0.017])
fig.colorbar(c2,cax=position,orientation='horizontal',format='%.1f',)
plt.savefig('ninoeof_jja_ts.png')
