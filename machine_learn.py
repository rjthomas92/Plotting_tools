import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from eofs.standard import Eof
import wget

nino               = xr.open_mfdataset('Data/ERSST/Summer/ersst.v5.*.nc',parallel=True, concat_dim="time",
                        combine="nested", data_vars='minimal', coords='minimal', compat='override')
nino_xa            = nino['sst'].sel(lat=slice(0,88))
nino_na            = nino_xa.values
tsz,e,ysz,xsz      = nino_xa.shape
yrsz               = int(tsz/3)
nino_ann           = nino_na.reshape(yrsz,3,ysz,xsz).mean(axis=1)
time               = str(nino['time'][0].values)
yrs                = np.arange(yrsz)+pd.to_datetime(time).year
lon, lat           = nino_xa['lon'].values, nino_xa['lat'].values

weights            = np.sqrt(np.cos((lat.reshape(ysz,1)@np.ones([1,xsz]))/180*np.pi))
solver             = Eof(nino_ann,weights=(-1)*weights)
eofs_cor           = solver.eofsAsCorrelation(neofs=10)
pcs                = solver.pcs(npcs=40,pcscaling=1)

nino_w               = xr.open_mfdataset('Data/ERSST/Winter/ersst.v5.*.nc',parallel=True, concat_dim="time",
                        combine="nested", data_vars='minimal', coords='minimal', compat='override')
nino_xa2            = nino_w['sst'].sel(lat=slice(-4,4),lon=slice(190,240))
nino_na2            = nino_xa2.values
tsz,e,ysz2,xsz2     = nino_xa2.shape
nino_ann2           = nino_na2.reshape(yrsz,3,ysz2,xsz2).mean(axis=1)
nino_ts             = nino_ann2 - nino_ann2.mean()
nino_ts             = np.nanmean(np.nanmean(nino_ts,axis=2),axis=1)

from sklearn.linear_model import LinearRegression # Linear regression
from sklearn.ensemble import RandomForestRegressor # Random Forest
from sklearn.svm import SVR # Support Vector
from sklearn.neural_network import MLPRegressor # Multi-layer Perceptron
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from scipy.stats import pearsonr
import warnings

warnings.simplefilter("ignore")

def get_metrics(y,y_pred):
    mse       = mean_squared_error(y,y_pred)
    rmse      = np.sqrt(mse)
    evr       = explained_variance_score(y,y_pred)
    cor, pval = pearsonr(y.ravel(),y_pred.ravel())
    if cor<0:
        pval  = 1.0
    return rmse, evr, cor, pval

# Length variables
tsz           = 23 # length of input time series
tsz_train     = 146 # Number of training points from initial time series

nin_sz        = 38 # Number of input time series
nsz           = 101 # Number of tests
scale_factors = np.arange(0,2.1,1.0) # Scale factor of training to predicted data
ssz           = len(scale_factors)
isz           = 2

# Initialisation
rmse_svr      = np.zeros([isz,ssz]); evr_svr=np.zeros([isz,ssz]); cor_svr=np.zeros([isz,ssz]); 
pval_svr      = np.zeros([isz,ssz]);

rmse_rfr      = np.zeros([isz,ssz]); evr_rfr=np.zeros([isz,ssz]); cor_rfr=np.zeros([isz,ssz]);
pval_rfr      = np.zeros([isz,ssz]);

rmse_lr       = np.zeros([isz,ssz]); evr_lr=np.zeros([isz,ssz]); cor_lr=np.zeros([isz,ssz]);
pval_lr       = np.zeros([isz,ssz]);

rmse_mlpr1    = np.zeros([isz,ssz]); evr_mlpr1=np.zeros([isz,ssz]); cor_mlpr1=np.zeros([isz,ssz]);
pval_mlpr1    = np.zeros([isz,ssz]);

rmse_mlpr2    = np.zeros([isz,ssz]); evr_mlpr2=np.zeros([isz,ssz]); cor_mlpr2=np.zeros([isz,ssz]);
pval_mlpr2    = np.zeros([isz,ssz]);

normlization  = 'standard'
x_sn_ratios   = [0.9,0.5] # S/N ratio in input data 1 means no noise in the input data (we only have one signal)
psz           = len(x_sn_ratios)
y_signals     = np.zeros([ssz,psz,nsz,tsz])
x_signals     = np.zeros([ssz,psz,nsz,tsz])


# Training and Prediction
for ind_x_sn_ratio,x_sn_ratio in enumerate(x_sn_ratios):
    ind_p           = ind_x_sn_ratio
    for ind_scale,scale_factor in enumerate(scale_factors):
        rmse_svr_   = np.zeros(nsz); evr_svr_=np.zeros(nsz); cor_svr_=np.zeros(nsz); pval_svr_= np.zeros(nsz);
        rmse_rfr_   = np.zeros(nsz); evr_rfr_=np.zeros(nsz); cor_rfr_=np.zeros(nsz); pval_rfr_= np.zeros(nsz);
        rmse_lr_    = np.zeros(nsz); evr_lr_=np.zeros(nsz); cor_lr_=np.zeros(nsz); pval_lr_=np.zeros(nsz);
        rmse_mlpr1_ = np.zeros(nsz); evr_mlpr1_=np.zeros(nsz); cor_mlpr1_=np.zeros(nsz);pval_mlpr1_= np.zeros(nsz);
        rmse_mlpr2_ = np.zeros(nsz); evr_mlpr2_=np.zeros(nsz); cor_mlpr2_=np.zeros(nsz);pval_mlpr2_= np.zeros(nsz);
        
        for n in range(nsz):
            print('x_sn_ratio=',x_sn_ratio,' scale_factor=',scale_factor, ' n=',n)
            ts           = np.arange(tsz)
            X            = pcs
            y            = nino_ts

            lr           = LinearRegression() # generation of instance
            lr.fit(X[:tsz_train,:], y[:tsz_train]) # training
            y_lr         = lr.predict(X) # prediction
            rmse_lr_[n], evr_lr_[n], cor_lr_[n], pval_lr_[n]             = get_metrics(y[tsz_train:],y_lr[tsz_train:])
            svr          = SVR()
            svr.fit(X[:tsz_train,:], y[:tsz_train])
            y_svr        = svr.predict(X)
            rmse_svr_[n], evr_svr_[n], cor_svr_[n], pval_svr_[n]         = get_metrics(y[tsz_train:],y_svr[tsz_train:])
            rfr          = RandomForestRegressor()
            rfr.fit(X[:tsz_train,:], y[:tsz_train])
            y_rfr        = rfr.predict(X)
            rmse_rfr_[n], evr_rfr_[n], cor_rfr_[n], pval_rfr_[n]         = get_metrics(y[tsz_train:],y_rfr[tsz_train:])
            mlpr1        = MLPRegressor(hidden_layer_sizes=(100))
            mlpr1.fit(X[:tsz_train,:], y[:tsz_train])
            y_mlpr1      = mlpr1.predict(X)
            rmse_mlpr1_[n], evr_mlpr1_[n], cor_mlpr1_[n], pval_mlpr1_[n] = get_metrics(y[tsz_train:],y_mlpr1[tsz_train:])
            mlpr2        = MLPRegressor(hidden_layer_sizes=(100,100))
            mlpr2.fit(X[:tsz_train,:], y[:tsz_train])
            y_mlpr2      = mlpr2.predict(X)
            rmse_mlpr2_[n], evr_mlpr2_[n], cor_mlpr2_[n], pval_mlpr2_[n] = get_metrics(y[tsz_train:],y_mlpr2[tsz_train:])

fig = plt.figure(figsize=(10,6))
plt.plot(yrs[146:],nino_ts[146:],'--k',label='Observed')
plt.plot(yrs[146:],y_lr[146:],label='Linear')
plt.plot(yrs[146:],y_svr[146:],label='Support Vector')
plt.plot(yrs[146:],y_rfr[146:],label='Random Forest')
plt.plot(yrs[146:],y_mlpr1[146:],label='MLPR1')
plt.plot(yrs[146:],y_mlpr2[146:],label='MLPR2')
plt.xlabel('Year')
plt.ylabel('SST Anomaly [oC]')
plt.title('Comparison of Observed and Predicted Nino3.4 Wintertime Anomaly')
plt.legend()

print(rmse_lr_[n], cor_lr_[n], pval_lr_[n])
print(rmse_svr_[n], cor_svr_[n], pval_svr_[n])
print(rmse_rfr_[n], cor_rfr_[n], pval_rfr_[n])
print(rmse_mlpr1_[n], cor_mlpr1_[n], pval_mlpr1_[n])
print(rmse_mlpr2_[n], cor_mlpr2_[n], pval_mlpr2_[n])
