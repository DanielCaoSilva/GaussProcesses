import pandas as pd
from pydap.client import open_url
import numpy as np


# https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/46211/46211h2022.nc
class DataGrab:

	base_data_set = r'https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/'
	raw_dataset = []
	data_url = []
	variable_keys = []
	columns_out = []

	def __init__(
				self,
				site_number='46221',
				year='2022',
				data_type='wave_height',
				debug=False):

		self.site = site_number
		self.year = year
		self.data_type = data_type
		self.debug = debug

	def get_urls(self):
		self.data_url.append(f'{self.base_data_set}{self.site}/{self.site}h{self.year}.nc')

	def open_urls(self):
		for d in self.data_url:
			self.raw_dataset.append(open_url(d, output_grid=False))

	def process_buoy_data(self):
		self.data_url.append(f'{self.base_data_set}{self.site}/{self.site}h{self.year}.nc')
		self.raw_dataset = open_url(self.data_url[0], output_grid=False)
		self.variable_keys = list(self.raw_dataset.keys())
		# columns_out = []
		total_wave_data = pd.DataFrame()
		for i in range(len(self.variable_keys)):
			if (i != 1) and (i != 2):
				self.columns_out.append(str(self.variable_keys[i]))
				total_wave_data[self.variable_keys[i]] = np.array(
						self.raw_dataset[self.variable_keys[i]][:].data,
						dtype="<f4") \
					.ravel()
				if self.debug:
					print(total_wave_data)
		return total_wave_data

	def grab_data(self):
		self.get_urls()
		rtn_data = pd.DataFrame(self.process_buoy_data(), columns=self.columns_out).astype(float)
		rtn_data.to_feather(f'../data/{self.site}_{self.year}_{self.data_type}.feather')
		rtn_data.to_csv(f'../data/{self.site}_{self.year}_{self.data_type}.csv')
		return rtn_data

	def get_vars(self):
		return self.variable_keys

	def close(self):
		# del self.raw_dataset
		# del self.data_url
		# del self.variable_keys
		# del self.columns_out
		self.raw_dataset = []
		self.data_url = []
		self.variable_keys = []
		self.columns_out = []


# Running Tests
# list_of_years = ['2018', '2019', '2020', '2021', '2022', '2021']
# for i in list_of_years:
# dg = DataGrab(year='9999', site_number='46221', data_type='wave_height', debug=False)
# print(dg.grab_data())
# dg.close()
# del dg
# dg = DataGrab(year=year_to_grab, site_number='46221', data_type='wave_height', debug=True)
# print(dg.grab_data())
# print("Grab urls")
# dg.get_urls()
# print("Open urls")
# dg.open_url()
# keys = list(dg.raw_dataset[0].keys())
# print(keys)
#
# df = pd.DataFrame()
# # print(
# # 	np.array(dg.raw_dataset[0][keys[0]][:].data, dtype="<f4").ravel()
# # )
#
# # 	# print(df)
# print(df)
# # print(dg.raw_dataset[0][keys[3:]].array[:])
#
# # print(pd.DataFrame(dg.raw_dataset[0].data[3]))#.data[0]))
# # print(pd.DataFrame(dg.raw_dataset[0][3][]))#.data[0]))
# # dg = DataGrab(year=["2022", "2021"])
# # wave_dat = dg.grab_data()
# # print(wave_dat)
