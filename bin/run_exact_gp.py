import torch
import gpytorch
import copy
from matplotlib import pyplot as plt
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from skimage.measure import block_reduce
from pathlib import Path
from datetime import datetime
import gc
import sys
import os
sys.path.append(Path(os.getcwd()).parent.__str__())
from src.utils import *
plt.style.use('classic')
set_gpytorch_settings(False)


# Reading data file and cleaning missing values
df = pd.read_feather(
	'../Data/feather/46221_9999_wave_height.feather')
parameters_wave = ['time', 'wave_height']
parameters_temp = ['time', 'sea_surface_temperature']
df_as_np = df \
	.loc[:, parameters_wave] \
	.astype(float) \
	.replace(
		to_replace=[999.0, 99.0, 9999.0], value=np.nan) \
	.to_numpy()
using_sk = block_reduce(
	df_as_np, block_size=(24, 1),
	func=np.mean).astype(float)
X = torch.tensor(using_sk[:-1, 0]).float().cuda()
y = torch.tensor(using_sk[:-1, 1]).float().cuda()
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

X = X[~torch.any(y.isnan(), dim=1)]
y = y[~torch.any(y.isnan(), dim=1)]
y = y.flatten()
X_old = X


# Helper functions
def scaler(
		a, X_old=X_old, center=True):
	if center is True:
		a = a - X_old.min(0).values
	return a / (X_old.max(0).values - X_old.min(0).values)


def add_new_kernel_term(
		original_kernel, new_kernel_term, operation):
	return str(original_kernel) + str(operation) + str(new_kernel_term)


# GP Model Declaration
class ExactGPModel(gpytorch.models.ExactGP):
	def __init__(self, train_x_, train_y_, likelihood, kernel):
		super(ExactGPModel, self).__init__(train_x_, train_y_, likelihood)
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = kernel

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions \
			.MultivariateNormal(mean_x, covar_x)


# Breath-Level Kernel Search Function - search_for_min_BIC()kkk
def search_for_min_BIC(
		possible_kernel_list, possible_kernel_operations,
		kernel_str_running_start, bic_values_list,
		data, scale, outer_count, initial_lr=0.01, epoch_iter=1000):
	trial_n = 0
	trial_hist = []
	potential_best_kernel = kernel_str_running_start
	start_bic = 999999
	# Iterate through all possible combinations of kernel terms and operations
	for kernel_ops_i, iter_operations in enumerate(possible_kernel_operations):
		for kernel_term_i, iter_terms in enumerate(possible_kernel_list):
			# If this is the first iteration, use the initial kernel string
			if (kernel_term_i == 0) and (kernel_ops_i == 0) and (outer_count == 0):
				kernel_str_current = kernel_str_running_start
				# kernel_str_current = add_new_kernel_term(
				# 	kernel_str_running_start, iter_terms, iter_operations)
			else:
				kernel_str_current = add_new_kernel_term(
					kernel_str_running_start, iter_terms, iter_operations)
			exact_gp_obj = TrainTestPlotSaveExactGP(
				ExactGPModel,
				kernel=kernel_str_current,
				train_x=data[0], train_y=data[1],
				test_x=data[2], test_y=data[3],
				scaler_min=scale[1], scaler_max=scale[0],
				num_iter=epoch_iter,
				lr=initial_lr,  # lr=0.0063, #lr=0.01,
				name=kernel_str_current,
				save_loss_values="save",
				use_scheduler=True)
			bic_at_current, hyper_vals = exact_gp_obj \
				.run_train_test_plot_kernel(
					set_xlim=[0.96, 1])
			if np.isnan(bic_at_current):
				del exact_gp_obj
				gc.enable()
				gc.collect()
				torch.cuda.empty_cache()
				print("There was a Nan in the BIC value, running again...")
				exact_gp_obj = TrainTestPlotSaveExactGP(
					ExactGPModel,
					kernel=kernel_str_current,
					train_x=data[0], train_y=data[1],
					test_x=data[2], test_y=data[3],
					scaler_min=scale[1], scaler_max=scale[0],
					num_iter=epoch_iter,
					lr=0.0063, #lr=0.01,
					name=kernel_str_current,
					save_loss_values="save",
					use_scheduler=True)
				bic_at_current, hyper_vals = exact_gp_obj \
					.run_train_test_plot_kernel(
						set_xlim=[0.96, 1])
			if (kernel_term_i == 0) and (kernel_ops_i == 0):
				start_bic = bic_at_current
			print(
				"Iterations Number(n): ", (outer_count, trial_n),
				"Learning Rate: ", initial_lr)
			print(
				"Kernel Structure: ", kernel_str_running_start,
				"\n BIC: ", start_bic, "(Start)")
			print(
				"Kernel Structure: ", potential_best_kernel,
				"\n BIC: ", bic_values_list[-1], " (Potential New Best)")
			print(
				"Kernel Structure: ", kernel_str_current,
				"\n BIC: ", bic_at_current, "(Current Trial)")
			trial_hist.append([
				(outer_count, trial_n), kernel_str_current,
				bic_at_current])
				# hyper_vals])
				#exact_gp_obj.kernel])
			if bic_at_current < bic_values_list[-1]:
				bic_values_list.append(bic_at_current)
				potential_best_kernel = kernel_str_current
				# kernel_str_running_start = kernel_str_current
				# del exact_gp_obj
				# gc.enable()
				# gc.collect()
				# torch.cuda.empty_cache()
				# return kernel_str_running_start, trial_hist
			trial_n += 1
			del exact_gp_obj
			gc.enable()
			gc.collect()
			torch.cuda.empty_cache()
	return kernel_str_running_start, trial_hist, potential_best_kernel


# Depth-Level search Function and save results
def save_results(search_params):
	full_path_history = []
	today_string = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
	full_path_history_string = "full_path_history_" + today_string + ".csv"
	try:
		for i in range(10):
			search_params["outer_count"] = i
			start_kernel, path_history, found_kernel = search_for_min_BIC(**search_params)
			full_path_history.append(path_history)
			if start_kernel == found_kernel:
				break
			search_params["kernel_str_running_start"] = found_kernel
	except:
		fph_df = pd.DataFrame(full_path_history)
		fph_df.to_csv(full_path_history_string)
		print("Threw exception, saving data...")
		print("Last kernel found: ", search_params["kernel_str_running_start"])
		print(fph_df)
	fph_df = pd.DataFrame(full_path_history)
	fph_df.to_csv(full_path_history_string)
	cleaned_fph_df = clean_csv_saves(full_path_history_string)
	print(cleaned_fph_df)
	print("Last kernel found: ", search_params["kernel_str_running_start"])
	return cleaned_fph_df


# Scale the time axis and log transform the Y-values
X = scaler(X, X_old)
y = y.log()

# max, min, and scale factor declaration
scaler_max = X_old.max(0).values.item()
scaler_min = X_old.min(0).values.item()
scale_factor = scaler_max - scaler_min
scaler_consts = [scaler_max, scaler_min, scale_factor]
temp_for_plotting = pd.Series(
	using_sk[:-1, 0] * 1e9, dtype='datetime64[ns]')
plt.plot(temp_for_plotting, using_sk[:-1, 1])
plt.xlabel("Time (epoch)")
plt.ylabel("Significant Wave Height (meters)")
plt.title(f'Significant wave height - after block reducing')
plt.show()

print(
	f'Scale Max: {scaler_max}\n '
	f'Scale Min: {scaler_min}\n '
	f'Scale Factor: {scale_factor}'
	f'Before Block Reduce: {df_as_np.shape}\n'
	f'After Block Reduce: {using_sk.shape}\n'
	f'Number of Nans: {np.count_nonzero(np.isnan(df_as_np))}\n'
	f'Start Time: {datetime.fromtimestamp(df_as_np[0, 0])}\n'
	f'End Time: {datetime.fromtimestamp(df_as_np[-1, 0])}\n'
	f'Number of Days: {df_as_np.shape[0] / 48}\n'
	f'Time Period (Days): {(df_as_np[-1, 0] - df_as_np[0, 0]) / 24 / 60 / 60}')

# Prediction range, training and test set define (14, 3, 365)
predict_days_out = 28
test_n = 2 * predict_days_out

# Split the data into train and test sets
# *contiguous means they are sitting next to each other in memory*
train_x = X[test_n:].contiguous().cuda()
train_y = y[test_n:].contiguous().cuda()
test_x = X[-test_n:].contiguous().cuda()
test_y = y[-test_n:].contiguous().cuda()

# Create a list of random starting indices for the subtest sets
n_total = train_x.shape[0]
idx_list = np.random.randint(
	low=n_total/2, high=n_total-test_n, size=1500)

# Generate the train_loader and train_dataset
train_loader, train_dataset, test_loader, test_dataset = create_train_loader_and_dataset(
	train_x, train_y, test_x, test_y)
data_compact = [
	train_x, train_y, test_x, test_y,
	train_loader, train_dataset,
	test_loader, test_dataset]

# List of possible Kernels operations
kernel_operations = ["+", "*"]

# List of possible Kernels terms
kernel_list = [
	# Varying Length Scales of the RBF Kernel
	"RQ",
	# Periodic Kernels of Varying Period constraints
	"Per_Arb", "Per_Year", "Per_Season", "Per_Month", "Per_Week"  # "Per_Unbounded"]
	# Random Fourier Features Kernel
	# "RFF",
	# Speciality Kernels
	"AR2", "Min",
	# Smoothing Kernels of the Matern class
	"RBF", "Mat_2.5", "Mat_1.5", "Mat_0.5",
]

# Initial Kernel Trial
kernel_str_running = "Per_Arb"

# Initializing empty list to record values
save_history = []
bic_values = [10000]
n = 0
initial_learning_rate = 0.01  # 0.0063 # initial learning rate
with_and_without_scheduler = [True]
search_parameters = {
	"possible_kernel_list": kernel_list,
	"possible_kernel_operations": kernel_operations,
	"kernel_str_running_start": kernel_str_running,
	"bic_values_list": bic_values,
	"data": data_compact,
	"scale": scaler_consts,
	"outer_count": 0,
	"initial_lr": initial_learning_rate,
	"epoch_iter": 1000
}

returned_results = save_results(search_parameters)
