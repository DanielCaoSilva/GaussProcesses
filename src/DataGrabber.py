import pandas as pd
from pydap.client import open_url
import numpy as np
from skimage.measure import block_reduce
from datetime import datetime


# DataGrab class: This class is used to grab data from the NOAA buoy data set
# get_urls: This function creates the url for the data set
# open_urls: This function opens the url and stores the data in a dictionary
# process_buoy_data: This function processes the data and stores it in a pandas dataframe
# grab_data: This function calls the other functions in the class and returns the data in a pandas dataframe
# get_vars: This function returns the variables in the data set
# close: This function closes the data set
class DataGrab:
	"""
	This class is used to grab data from the NOAA buoy data set
	"""

	base_data_set = r'https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/'
	raw_dataset = []
	data_url = []
	variable_keys = []
	columns_out = []
	rtn_data = pd.DataFrame()
	file_name = ''

	def __init__(
				self,
				site_number='46221',
				year='2022',  # When year is 9999, it grabs the last 10 years to date
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
						dtype="<f4").ravel()
				if self.debug:
					print(total_wave_data)
		return total_wave_data

	def grab_data(self, return_data=False, save_as="feather"):
		self.get_urls()
		self.rtn_data = pd.DataFrame(self.process_buoy_data(), columns=self.columns_out).astype(float)
		self.file_name = f'../Data/{self.site}_{self.year}_{self.data_type}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
		if save_as == "feather":
			self.rtn_data.to_feather(f'{self.file_name}.feather')
		if save_as == "csv":
			self.rtn_data.to_csv(f'{self.file_name}.csv')
		if return_data:
			return self.rtn_data

	def block_reduce_data(self, block_size=24):
		parameters_wave = ['time', self.data_type]
		df_as_numpy = self.rtn_data \
			.loc[:, parameters_wave] \
			.astype(float) \
			.replace(
				to_replace=[999.0, 99.0, 9999.0],
				value=np.nan) \
			.to_numpy()
		using_sk = block_reduce(
			df_as_numpy, block_size=(block_size, 1),
			func=np.mean).astype(float)
		using_sk_df = pd.DataFrame(
			using_sk,
			columns=parameters_wave)
		using_sk_df.to_feather(f'{self.file_name}_block_reduce.feather')
		return using_sk

	def grab_data_block_reduce(self, return_data=False, save_as="feather", block_size=24):
		print("Starting Data Collection...")
		self.grab_data(return_data=return_data, save_as=save_as)
		print("Data Collection Complete.  Starting Block Reduce...")
		return self.block_reduce_data(block_size=block_size)

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

# https://dods.ndbc.noaa.gov/thredds/dodsC/data/stdmet/46211/46211h2022.nc
# <gml:beginPosition>2012-09-30T23:55:00Z</gml:beginPosition>
# <gml:endPosition>2023-03-30T22:26:00Z</gml:endPosition>
# year 9999 is starting on Sept 30, 2012 to current date