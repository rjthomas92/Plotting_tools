def show_ini(in_file):
    import matplotlib.pyplot as plt
    import xarray as xr
  
    ocean_file = (in_file)
    ocean = xr.open_dataset(ocean_file)
    
    temp_ini = ocean.temp.isel(ocean_time=0,s_rho=-1)
    salt_ini = ocean.salt.isel(ocean_time=0,s_rho=-1)
  
    tempmin = int(np.min(temp_ini))
    tempmax = int(np.max(temp_ini))
    saltmin = int(np.min(salt_ini))
    saltmax = int(np.max(salt_ini))

    lon_rho = ocean.lon_rho.isel(eta_rho=0)
    lat_rho = ocean.lat_rho.isel(xi_rho=0)
    lon_u = ocean.lon_u.isel(eta_u=0)
    lat_u = ocean.lat_u.isel(xi_u=0)
    mask_u = ocean.mask_u.isel()


    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(18, 6))
    fig.suptitle('Initial Conditions')
  
    ax1.contour(lon_u,lat_u,mask_u,cmap='binary')
    sh1 = ax1.contourf(lon_rho,lat_rho,temp_ini,cmap='RdBu_r',extend='both',levels=np.linspace(tempmin,tempmax))
    fig.colorbar(sh1, ax=(ax1), orientation='vertical',ticks=np.arange(tempmin,tempmax+1,2))
    ax1.set_title("Initial SST")
    ax1.set_ylabel('Latitude')

    ax2.contour(lon_u,lat_u,mask_u,cmap='binary')
    sh2 = ax2.contourf(lon_rho,lat_rho,salt_ini,cmap='RdBu_r',extend='both',levels=np.linspace(varmin,varmax))
    fig.colorbar(sh2, ax=(ax2), orientation='vertical',ticks=np.arange(saltmin,saltmax+0.1,0.2))
    ax2.set_title("Initial Salinity")
    fig.supxlabel('Longitude')
  
    plt.show()
    plt.savefig('ini.png')
    plt.clf()
