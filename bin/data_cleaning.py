import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

# list_of_file = glob.glob('../data/*.csv')
# out_df = pd.DataFrame()
# for file in list_of_file:
# 	temp_df = pd.read_csv(file, low_memory=False)
# 	out_df = pd.concat([out_df, temp_df])
# out_df.to_csv('../data/combined_data.csv')

def moving_average(data, window_size):
	window = np.ones(int(window_size)) / float(window_size)
	return np.convolve(data, window, 'same')



# pd.read_feather('../data/feather/combined/combined_data.feather')
df = pd.read_feather('../data/feather/46221_9999_wave_height.feather')
df_as_np = df.loc[:, 'sea_surface_temperature'].astype(float).to_numpy()
print(len(df_as_np))
print(df_as_np)
smaller_df = moving_average(df_as_np, window_size=24)
print(len(smaller_df))
print(smaller_df)
plt.plot(df_as_np)
plt.plot(smaller_df)