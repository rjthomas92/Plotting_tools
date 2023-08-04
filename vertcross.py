def vertcross(in_file,tidx,tfdx,irange,j)
  nc = netCDF4.Dataset(in_file)
  eta = nc.variables['zeta'][tidx, :, :]
  # calculate the 3D field of z values (vertical coordinate) at this time step
  
  mask = nc.variables['mask_rho'][:]
  s = nc.variables['s_rho'][:]
  a = nc.variables['theta_s'][:]
  b = nc.variables['theta_b'][:]
  depth_c = nc.variables['hc'][:]
  depth = nc.variables['h'][:,:]
  lon_rho = nc.variables['lon_rho'][:,:]
  lat_rho = nc.variables['lat_rho'][:,:]
  lon_u = nc.variables['lon_u'][:,:]
  lat_u = nc.variables['lat_u'][:,:]
  lon_v = nc.variables['lon_v'][:,:]
  lat_v = nc.variables['lat_v'][:,:]
  
  lon3d = np.ones((32,1,1))*lon_rho
  
  temp_ini = nc.variables['temp'][tidx, :, :, :]
  salt_ini = nc.variables['salt'][tidx, :, :, :]
  
  #u = nc.variables['u'][100:200, :, :, :]
  #u_ini = nc.variables['u'][100, :, :, :]
  #v = nc.variables['v'][100:150, :, :, :]
  #v_ini = nc.variables['v'][100, :, :, :]
  #w = nc.variables['w'][tidx, :, :, :]
  #w_ini = nc.variables['w'][100, :, :, :]
  
  C = (1-b)*np.sinh(a*s)/np.sinh(a) + b*[np.tanh(a*(s+0.5))/(2*np.tanh(0.5*a)) - 0.5]
  
  C.shape = (np.size(C), 1, 1)
  s.shape = (np.size(s), 1, 1)
  
  z = eta*(1+s) + depth_c*s + (depth-depth_c)*C
  
  lon3d = np.ones((32,1,1))*lon_rho
  
  fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
  
  temp_ini = nc.variables['temp'][0, :, :, :]
  temp_diff = temp-temp_ini
  
  jval=35
  irange=range(80,100)
  
  sh1 = ax1.pcolormesh(lon3d[:,jval,irange],z[:,jval,irange],temp_ini[:,jval,irange],cmap='RdBu_r',vmin=15)
  ax1.set_title('Temperature Anomaly Cross Section')
  #ax1.colorbar(extend='both')
  #ax1.set_clim(15,30)
  fig.colorbar(sh1, ax=(ax1), orientation='vertical',ticks=np.arange(15,31,2))
  ax1.set_ylabel('Depth [m]')
  #ax1.set_xlabel('Longitude')
  ax1.set_ylim([-600,10])
  
  sh2 = ax2.pcolormesh(lon3d[:,jval,irange],z[:,jval,irange],temp_diff[:,jval,irange],cmap='RdBu_r')
  fig.colorbar(sh2, ax=(ax2), orientation='vertical',ticks=np.arange(-4,4,2))
  #ax1.colorbar(extend='both')
  #ax1.set_clim(15,30)
  ax2.set_ylabel('Depth [m]')
  ax2.set_xlabel('Longitude')
  ax2.set_ylim([-600,10])
