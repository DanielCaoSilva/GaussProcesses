import pandas as pd
from pydap.client import open_url


class DataGrab:
	base_data_set = r'https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/'
	raw_dataset = []
	data_url = []

	def __init__(self, site_number='46221', year=None, data_type='wave_height'):
		if year is None:
			year = ['2022']
		self.site = site_number
		self.year = year
		self.data_type = data_type

	def get_urls(self):
		#if self.year is list:
		for y in self.year:
			self.data_url.append(f'{self.base_data_set}{self.site}/{self.site}h{y}.nc')
		#else:
			#self.data_url[0] = f'{self.base_data_set}{self.site}/{self.site}h{self.year}.nc'

	def open_url(self):
		for d in self.data_url:
			self.raw_dataset.append(open_url(d))

	def process_data_wave_height(self):
		total_wave_data = pd.DataFrame()
		for r in self.raw_dataset:
			wave_height = r[self.data_type].data[0]
			time_ = r[self.data_type].data[1]
			wave_data = pd.DataFrame(time_, columns=['Time'])
			wave_data[self.data_type] = wave_height
			wave_data.set_index('Time')
			total_wave_data = pd.concat([total_wave_data, wave_data])
		return total_wave_data

	def grab_data(self):
		self.get_urls()
		self.open_url()
		return self.process_data_wave_height()


# dg = DataGrab(year=["2022", "2021"])
# wave_dat = dg.grab_data()
# print(wave_dat)



# Old stuff - Trying to use an api to read from pydap
# from pydap.client import open_url
# import netCDF4, pydap, urllib
# import pylab, matplotlib
# import numpy as np
# import pandas as pd
# #from opendap import opendap
#
# # dataset = open_url(
# # 	'https://dods.ndbc.noaa.gov/thredds/catalog/data/stdmet/14043/catalog.html?dataset=data/stdmet/14043/14043h9999.nc')
# class BuoyStation:
#
# 	public_key = '''
# mQGiBEcx+2IRBACiC0Rp/8fHy9PEHxf4f4H312AVPoxtYEfjXgBy7hPA/cHiltgs
# l5UyF6sHZlGxvJ44vuflztTO8yQejsicnMt4pWs44W0pgusERQ9yxkZzxfxkpH2r
# 1KAMa49Py9NEyP6uhjYC9iYVncuCc+YyHWdek5eCVBrHMGBnHqdDOFcDswCg3sNb
# 9+mFeVoKVZz78L4oTmlwrocEAIkXPz+N004Nj2YXo+KtW4aoYZITwVUCsM8AG/PI
# PR/W7gpkaFx87cAA/7O5Z6gTD0EGP0qQ4/9mzpci4HELjgYXd9RQvubT50sTW9nd
# 1nSgLSsxV+huTBwbofJ3hxxvgSyr8ZjQ2nflHjMcI1is8Dzzl+Q1b0Zb9So90JNK
# 6DSBA/wNlQzVweimpi8VA1oNSEm4ZCIYFxAlXiFIMkW/E93q+eGeQwuiHUolzIlP
# JW0t5Y+d2JUcoInyLSQtY5xWBNoUIwbbIpKro6b0FGCxkw5NFHkLHsESR9yn7tRy
# F6NqR0LnXpfuJbHcpHmhoE1w0mbqSlzQM7dTUigKgj3KLpx/XbQ3T1BlTkRBUCBT
# ZWN1cml0eSAoT1BlTkRBUCwgSW5jLikgPHNlY3VyaXR5QG9wZW5kYXAub3JnPohg
# BBMRAgAgBQJHMftiAhsDBgsJCAcDAgQVAggDBBYCAwECHgECF4AACgkQFS24nXN8
# JMQ98QCgx4LnDPCUj5SEcIaYaHGoNa3sR5UAnjopw3gT8pUZx97Vu8llU3zaFRMl
# iEYEEBECAAYFAkcx/0oACgkQaCA8XAomTJHujQCfeofPSX+0lj+VR1ZAn84oI2QA
# vNEAn1RhtS3pe8i1uIuV5HSus2McBep8uQINBEcx+2IQCACxq8iJRBYOgmoLRS3+
# lsM005gB9ApcCQxTTATtY/b9v+lXj5DbuNfqKDbgJkJ1dqMBNFfqKOZM8kgCWNLl
# S79pF7JgOjeT8nhZOfnLSO73aDnphUwq9D3gJsfizDcJscYFUXL+9YLOesGLBlV6
# acXWgdcgDjDw8EagXvy9xHJRN8CBBQ+cBFYoBuY2JoIx2YhLjYP4f7TK9RKdGaVr
# /2BLLCSeSP92PGM/7YLEk+5a6/VNda4uTC4W22toAdhP8LT2+nYmKCb0Q9FtIVDd
# kmtSmdmIZju4TYdYZI0GvkdJq1+GRzyoK+nlkQdPmWIze6KhASS5Zy/EeWJryr0B
# 89Y3AAMFCAClt8arTKMWA8Q8SlT6GxAGYlMImP4jbjaEgWDtdzyYcxt9YoNQnqWR
# DBd0NRTz3jI6L/sauFyW91j3SkbhAIw7BDLx6dZbq4cuJ68Fkw0gvHD7/QUzodpb
# JXEeo9IZKHt2QlvK7MreymZBYMiLl+AUFtQNPmCKS1fPCelx5/Gk0RCoVWeaMpzv
# ungi4CPW6ugd3J1MjOqsOUIJvO+KENm+t/HEfgNxvFmQ8tjvwWJSINapPOIpjMQl
# lZ0FAj5b7/I7IGMODokzXKEJLJfdth97iTrF8ZB6wx8woed8C1ndCbMzNxYdmzHO
# fAE0+hj4bhp+mQCUWr/mtIS0osEfmQf5iEkEGBECAAkFAkcx+2ICGwwACgkQFS24
# nXN8JMRa8QCgp/hM20XGHtw/+Zy5VT4DO7rtrDsAnA9xE8lJFjFv5bM5aLo7x8Eu
# abbs
# =f6nX
# '''
# 	data_url_base = r'https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/'
# 	#station_string = r'46268/46268/h2022.nc'
# 	station = r'46368'
# 	year = r'2022'
# 	station_string = station+'/'+station+'/h'+'.nc'
# #data_url = 'https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/14043/14043h9999.nc'
# 	data_url = 'https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/46221/46221h2022.nc'
# #data_url = 'https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/14043/14043h9999.nc'#?time[0:1:1],latitude[0:1:0],wave_height[0:1:10][0:1:0][0:1:0]'
# #dataset = open_url(data_url)
# 	def __init__(self, station_number, yr):
# 		self.station_string = station_number+'/'+station_number+'/h'+'.nc'
# 		self.data_url = self.data_url_base+self.station_string
# 		self.dataset = open_url(self.data_url, output_grid=False)
#
# 	def grab_data(self):  # stations of importance: 46221, 46368, 46025, ICAC1
# 		wvht = self.dataset['wave_height']
# 		time = self.dataset['time']
# 		wvht_grid = wvht[:, :, :]
# 		return wvht_grid.data, time.data
# #wave_height = \
#
# 	#data_columns = list(dataset.keys())#[1:100])#['wave_height']
#
#
#
# #print(wave_height.data)
#
# # class OceanBuoyData:
# # 	#dataset = open_url('https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/14043/14043h9999.nc?time[0:1:5705],latitude[0:1:0],longitude[0:1:0],wind_dir[0:1:0][0:1:0][0:1:0],wind_spd[0:1:0][0:1:0][0:1:0],gust[0:1:0][0:1:0][0:1:0],wave_height[0:1:0][0:1:0][0:1:0],dominant_wpd[0:1:0][0:1:0][0:1:0],average_wpd[0:1:0][0:1:0][0:1:0],mean_wave_dir[0:1:0][0:1:0][0:1:0],air_pressure[0:1:0][0:1:0][0:1:0],air_temperature[0:1:0][0:1:0][0:1:0],sea_surface_temperature[0:1:0][0:1:0][0:1:0],dewpt_temperature[0:1:0][0:1:0][0:1:0],visibility[0:1:0][0:1:0][0:1:0],water_level[0:1:0][0:1:0][0:1:0]')
# # 	def __int__(self):
