from skimage.measure import block_reduce
from pathlib import Path
from datetime import datetime
import gc
import sys
import os
# sys.path.append(Path(os.getcwd()).parent.__str__())
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

# Convert to torch tensors
X = torch\
	.tensor(using_sk[:-1, 0])\
	.float()\
	.cuda()
y = torch\
	.tensor(using_sk[:-1, 1])\
	.float()\
	.cuda()
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

X = X[~torch.any(y.isnan(), dim=1)]
y = y[~torch.any(y.isnan(), dim=1)]
y = y.flatten()
X_old = X


# Helper functions
def scaler(
		a,
		X_old=X_old,
		center=True):
	if center is True:
		a = a - X_old.min(0).values
	return a / (X_old.max(0).values - X_old.min(0).values)


def add_new_kernel_term(
		original_kernel, new_kernel_term, operation):
	return str(original_kernel) + str(operation) + str(new_kernel_term)


# GP Model Declaration
class ExactGPModel(gpytorch.models.ExactGP):
	def __init__(
			self,
			train_x_, train_y_,
			likelihood, kernel):
		super(ExactGPModel, self).__init__(train_x_, train_y_, likelihood)
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = kernel

	def forward(
			self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions \
			.MultivariateNormal(mean_x, covar_x)


# Scale the time axis and log transform the Y-values
X = scaler(X, X_old)
y = y.log()

# max, min, and scale factor declaration
scaler_max = X_old.max(0).values.item()
scaler_min = X_old.min(0).values.item()
scale_factor = scaler_max - scaler_min
scaler_consts = [scaler_max, scaler_min, scale_factor]

# Plot the block reduced data set
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
	f'Scale Factor: {scale_factor}\n '
	f'Before Block Reduce: {df_as_np.shape}\n'
	f'After Block Reduce: {using_sk.shape}\n'
	f'Number of Nans: {np.count_nonzero(np.isnan(df_as_np))}\n'
	f'Start Time: {datetime.fromtimestamp(df_as_np[0, 0])}\n'
	f'End Time: {datetime.fromtimestamp(df_as_np[-1, 0])}\n'
	f'Number of Days: {df_as_np.shape[0] / 48}\n'
	f'Time Period (Days): {(df_as_np[-1, 0] - df_as_np[0, 0]) / 24 / 60 / 60}\n ')

# Prediction range, training and test set define (14, 3, 365)
predict_days_out = 3
test_n = 2 * predict_days_out

# Split the data into train and test sets
# *contiguous means they are sitting next to each other in memory*
train_x = X[test_n:].contiguous().cuda()
train_y = y[test_n:].contiguous().cuda()
test_x = X[-test_n:].contiguous().cuda()
test_y = y[-test_n:].contiguous().cuda()

# Create a list of random starting indices for the subtest sets
n_total = train_x.shape[0]
np.random.seed(2023)
idx_list = np.random.randint(
	low=n_total/2,
	high=n_total-test_n,
	size=100)

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
	# Periodic Kernels of Varying Period constraints
	"Per_Arb", "Per_Year", "Per_Season", "Per_Month", "Per_Week",
	# Random Fourier Features Kernel
	"RFF",
	# Varying Length Scales of the RBF Kernel
	"RQ",
	# Speciality Kernels
	"AR2", "Min",
	# Smoothing Kernels of the Matern class
	"RBF", "Mat_2.5", "Mat_1.5", "Mat_0.5",
]

# Initial Kernel Trial
kernel_str_running = "AR2*RFF"
# kernel_str_running = "RBF+AR2*Per_Year*RBF*Mat_1.5"

parameter_input = {
	"model_cls": ExactGPModel,
	"kernel": kernel_str_running,
	"train_x": data_compact[0],
	"train_y": data_compact[1],
	"test_x": data_compact[2],
	"test_y": data_compact[3],
	"scaler_min": scaler_consts[1],
	"scaler_max": scaler_consts[0],
	"num_iter": 1000,
	"lr": 0.01,
	"name": kernel_str_running,
	"save_loss_values": "save",
	"use_scheduler": True,
}


def run_the_model(
		input_parameters,
		index_list,
		return_hyper_values=False):
	exact_gp = TrainTestPlotSaveExactGP(**input_parameters)
	bic_current_value, hyper_values = exact_gp\
		.run_train_test_plot_kernel(
			set_xlim=[0.96, 1],
			show_plots=True)
	print("Running Steps Ahead Error...")
	err_list = exact_gp\
		.steps_ahead_error(
			idx_list=index_list,
			predict_ahead=predict_days_out)
	plt.scatter(index_list, err_list)
	plt.xlabel("Index")
	plt.ylabel("Error")
	plt.title(
		f'CV Random Split: Index vs Error '
		f'({input_parameters["kernel"]}: RMSE={np.nanmean(err_list):.3f})')
	plt.show()
	print("BIC", bic_current_value)
	print("Avg Error", np.nanmean(err_list))
	print("Hyper Values", hyper_values)
	# print("Error List", err_list)
	del exact_gp
	if return_hyper_values:
		return bic_current_value, np.nanmean(err_list), hyper_values
	else:
		return bic_current_value, np.nanmean(err_list)


def test_steps_ahead_error(
		parameter_input_dictionary,
		seeded_idx_list):
	with_cv = []
	past_trials = pd.read_csv("./../Past_Trials/full_results/cleaned_all_trials.csv")
	past_trials.sort_values(by="BIC", inplace=True)
	for index, row in past_trials[0:10].iterrows():
		print(row["Kernel"])
		parameter_input["kernel"] = str(row["Kernel"]).replace("'", "")
		print("SAVED BIC: ", row["BIC"])
		b, e = run_the_model(parameter_input_dictionary, seeded_idx_list)
		with_cv.append([row["Kernel"], b, e])
	with_cv_df = pd.DataFrame(with_cv)
	with_cv_df.to_csv("top_10_trials_with_cv.csv")
	print(with_cv_df)


def run_list_of_models(
		param_in_dict,
		seeded_index_list,
		list_of_models_to_try,
		file_name="model_results.csv"):
	model_results_output = []
	for try_model in list_of_models_to_try:
		param_in_dict["kernel"] = try_model
		param_in_dict["name"] = try_model
		b, e, h = run_the_model(
			param_in_dict,
			seeded_index_list,
			return_hyper_values=True)
		model_results_output.append([try_model, b, e, h])
		gc.enable()
		gc.collect()
		torch.cuda.empty_cache()
	model_results_df = pd.DataFrame(model_results_output)
	model_results_df.to_csv(file_name)
	return model_results_df


same_model_again_list = ["RBF+AR2+Mat_2.5", "RQ"]
same_model_results = run_list_of_models(
	parameter_input,
	idx_list,
	same_model_again_list,
	file_name="same_model_results_2.csv")
print(same_model_results)

np.random.seed(2023)
idx_list = np.random.randint(
	low=n_total/2,
	high=n_total-test_n,
	size=1000)

same_model_again_list = ["RBF+AR2+Mat_2.5", "RQ"]
same_model_results = run_list_of_models(
	parameter_input,
	idx_list,
	same_model_again_list,
	file_name="same_model_results_3.csv")
print(same_model_results)

# base_models = run_list_of_models(parameter_input, idx_list, kernel_list)
# pt_df = pd.read_csv("./../Past_Trials/full_results/cleaned_all_trials.csv")
# pt_df.sort_values(by="BIC", inplace=True)
# pt_df = pt_df.loc[pt_df["BIC"] < 0]
# past_kernels_list = list(
# 	pt_df
# 	.loc[:, "Kernel"]
# 	.apply(lambda x: str(x).replace("'", "")))
# found_models = run_list_of_models(
# 	parameter_input, idx_list, past_kernels_list, file_name="previous_trials_model_results_2.csv")
# print(base_models)
# print(found_models)

# run_the_model(parameter_input, idx_list)
# gc.enable()
# gc.collect()
# torch.cuda.empty_cache()
# parameter_input["kernel"] = "RFF*AR2"
# parameter_input["name"] = "RFF*AR2"
# run_the_model(parameter_input, idx_list)
# 0.06681766470428556


