import pandas as pd
import glob

list_of_file = glob.glob('../data/*.csv')
out_df = pd.DataFrame()
for file in list_of_file:
	temp_df = pd.read_csv(file, low_memory=False)
	out_df = pd.concat([out_df, temp_df])
out_df.to_csv('../data/combined_data.csv')