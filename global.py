# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 08:13:07 2019

@author: ZongrAx
"""
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


#nc_colm = Dataset('CoLMCRU_2D_Fluxes_2001-07.nc')
#nc_clm = Dataset('CLMCRU_2D_Fluxes_2001-07.nc')
a = Dataset(r'C:\Users\adminX\Desktop\old_op\old_op_2D_Fluxes_2003-06.nc')
b = Dataset(r'C:\Users\adminX\Desktop\old_op\old_op_2D_Fluxes_2003-07.nc')
c = Dataset(r'C:\Users\adminX\Desktop\old_op\old_op_2D_Fluxes_2003-08.nc')
a1 = Dataset(r'C:\Users\adminX\Desktop\new_op\new_op_2D_Fluxes_2003-06.nc')
b1 = Dataset(r'C:\Users\adminX\Desktop\new_op\new_op_2D_Fluxes_2003-07.nc')
c1 = Dataset(r'C:\Users\adminX\Desktop\new_op\new_op_2D_Fluxes_2003-08.nc')
albdo1 = a['f_fevpa'][:][:]
albdo2 = b['f_fevpa'][:][:]
albdo3 = c['f_fevpa'][:][:]
albdn1 = a1['f_fevpa'][:][:]
albdn2 = b1['f_fevpa'][:][:]
albdn3 = c1['f_fevpa'][:][:]

albdo = (albdo1 + albdo2 + albdo3)/3
albdn = (albdn1 + albdn2 + albdn3)/3
assim[360][720] = albdo
for i in range(360):
    for j in range(720):
        if albdo[i][j] != 0:
            assim[i][j] = (albdn[i][j] - albdo[i][j]) / albdo[i][j]
        else:
            assim[i][j] = albdn[i][j]
lon_o = a['lon'][:]
lat_o = a['lat'][:]
lon_n = a1['lon'][:]
lat_n = a1['lat'][:]

fig = plt.figure(figsize=(10, 7))
ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())

ax1.coastlines()
ax1.set_global()
# m.drawcoastlines()
# m.drawlsmask()
# m.drawcountries()
gl = ax1.gridlines(ylocs=np.arange(-90, 90+30, 30), xlocs=np.arange(-180,
                                                                    180+60, 60), draw_labels=True, linestyle='--', alpha=0.7)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
#lons, lats = np.meshgrid(lon, lat)
#x, y = m(lons, lats)
x_colm, y_colm = np.meshgrid(lon_o, lat_o)
# minval,maxval=int(np.amin(albd_colm)),int(np.amax(albd_colm))+1
cs = ax1.pcolormesh(x_colm, y_colm, albdo, cmap=plt.cm.get_cmap('hot_r'))
plt.colorbar(cs)
plt.title('not updata vis (%)')
# plt.savefig('O_DJF.png')
#cs = m.contourf(x, y, assim, clevs, cmap=cm.s3pcpn)
#cbar = m.colorbar(cs, location='bottom', pad="10%")

ax2 = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())

ax2.coastlines()
ax2.set_global()

gl = ax2.gridlines(ylocs=np.arange(-90, 90+30, 30), xlocs=np.arange(-180,
                                                                    180+60, 60), draw_labels=True, linestyle='--', alpha=0.7)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

x_clm, y_clm = np.meshgrid(lon_n, lat_n)
plt.title('updata vis(%)')
cs2 = ax2.pcolormesh(x_clm, y_clm, albdn, cmap=plt.cm.get_cmap('hot_r'))
plt.colorbar(cs2)
plt.title('updata vis (%)')
# plt.savefig('N_DJF.png')

ax3 = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())

ax3.coastlines()
ax3.set_global()
gl = ax3.gridlines(ylocs=np.arange(-90, 90+30, 30), xlocs=np.arange(-180,
                                                                    180+60, 60), draw_labels=True, linestyle='--', alpha=0.7)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
x, y = np.meshgrid(lon_n, lat_n)
cs3 = ax3.pcolormesh(x, y, assim, cmap=plt.cm.get_cmap('jet'))
plt.colorbar(cs3)
plt.savefig('DJF.png')

plt.show()
