def vertprof(in_file,tidx,tfdx,i,j):
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
  
  fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,12))
  
  sst1post = 0
  n=0
  for t in range (15,40):
      sst1post = nc.variables['temp'][t, :,j, i] + sst1post
      n+=1
      
  sst1post = sst1post/n
  
  ax1.plot(temp_ini,z[:,j,i])
  ax1.plot(sst1post,z[:,j,i])
  ax1.set_ylim(-600,0)
  ax1.set_xlim(15,30)
  plt.suptitle('Vertical Temperature Anomaly')
  
  salt1post = 0
  n=0
  for t in range (15,40):
      salt1post = nc.variables['salt'][t, :,40, 70] + salt1post     
      n+=1
          
  salt1post = salt1post/n
  
  ax2.plot(salt_ini,z[:,35,85])
  ax2.plot(salt1post,z[:,35,85])
  ax2.set_ylim(-600,0)
  ax2.set_xlim(15,31)
