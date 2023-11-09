# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Pacotes
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

# Caminho para meu arquivo netCDF com concentração de ozônio
path = r"C:\ENS410064\dados\netCDF4\BRAIN_ClippedCONC_O3_2019_07_02_11_to_2019_12_30_23.nc"

# Abrir arquivo netCDF4
data = nc.Dataset(path)
data

# Extraindo dados de ozônio
o3 = data['O3'][:]
lat = data['LAT'][:]
lon = data['LON'][:]
tflag = data['TFLAG'][:]

# Figura de um ponto em todos os pontos
fig,ax = plt.subplots()
ax.plot(o3[:,0,50,50])

# Figura de um tempo e todo o espaço
fig2,ax2 = plt.subplots()
ax2.pcolor(lon,lat,o3[0,0,:,:])

# Figura de um tempo e todo o espaço (média anual)
fig3,ax3 = plt.subplots()
ax3.pcolor(lon,lat,np.mean(o3[:,0,:,:], axis = 0))
shp = gpd.read_file(r"C:\ENS410064\dados\brutos\analiseEspacialAula04\BR_Municipios_2022\BR_Municipios_2022.shp")
shp.boundary.plot(ax=ax3)
