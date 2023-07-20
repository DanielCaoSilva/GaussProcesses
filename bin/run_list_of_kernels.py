import numpy as np
from skimage.measure import block_reduce
from datetime import datetime
import gc
import sys
import os
from pathlib import Path
# print(os.getcwd())
# if Path(os.getcwd()).__str__() not in sys.path:
#     sys.path.append(Path(os.getcwd()).parent.__str__())
from src.utils import *

# plt.style.use('bmh')
# plt.style.use('dark_background')
# plt.style.use('Solarize_Light2')
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
        to_replace=[999.0, 99.0, 9999.0],
        value=np.nan) \
    .to_numpy()
using_sk = block_reduce(
    df_as_np, block_size=(24, 1),
    func=np.mean).astype(float)

# Convert to torch tensors
X = torch \
    .tensor(using_sk[:-1, 0]) \
    .float() \
    .cuda()
y = torch \
    .tensor(using_sk[:-1, 1]) \
    .float() \
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
# plt.plot(temp_for_plotting, using_sk[:-1, 1])
# plt.xlabel("Time (epoch)")
# plt.ylabel("Significant Wave Height (meters)")
# plt.title(f'Significant wave height - after block reducing')
# plt.show()

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
# train_x = X[test_n:].cuda()
# train_y = y[test_n:].cuda()
# test_x = X[-test_n:].cuda()
# test_y = y[-test_n:].cuda()
train_x = X[test_n:].contiguous().cuda()
train_y = y[test_n:].contiguous().cuda()
test_x = X[-test_n:].contiguous().cuda()
test_y = y[-test_n:].contiguous().cuda()
#
# # Forecasting beyond horizon
# test_future_15 = torch.cat((X[-test_n:], (X[1:(test_n*5)]+1)), dim=0).contiguous().cuda()
# test_future_90 = torch.cat((X[-test_n:], (X[1:(test_n*30)]+1)), dim=0).contiguous().cuda()
# train_x = X[test_n:].cuda()
# train_y = y[test_n:].cuda()
# test_x = X[-test_n:].cuda()
# test_y = y[-test_n:].cuda()

# Forecasting beyond horizon
# test_future_15 = torch.cat((X[-test_n:], (X[1:(test_n*5)]+1)), dim=0).cuda()
# test_future_90 = torch.cat((X[-test_n:], (X[1:(test_n*30)]+1)), dim=0).cuda()
# print(test_future_15)
# print(test_future_90)
# print(test_x)

# Create a list of random starting indices for the subtest sets
n_total = train_x.shape[0]
np.random.seed(2023)
torch.manual_seed(2023)
idx_list = np.random.randint(
    low=n_total / 2,
    high=n_total - test_n,
    size=1000)


def make_idx_list(
        training_set_size,
        size_of_artificial_test_set,
        size_of_partitions=1000, seed=2023):
    np.random.seed(seed)
    return np.random.randint(
        low=training_set_size / 2,
        high=training_set_size - size_of_artificial_test_set,
        size=size_of_partitions)


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
    "forecast_over_this_horizon": [4, 10], #None, #[test_future_15, test_future_90],
    "index_list_for_training_split": idx_list,
    "predict_ahead_this_many_steps": test_n,
}


# Runs A single instance of the GP model based on the input parameters, one of which contain the kernel string
# Extra Parameters:
#   return_hyper_values:=False, return the actual hyperparameters of the TrainTestPlotSaveExactGP object
#   run_steps_ahead_error=False, run the steps_ahead_error() method or not
#       from the TrainTestPlotSaveExactGP object
# Order of execution of the model: test, save_model_state, get_bic, plot, eval_prediction, steps_ahead_error
def run_the_model(
        input_parameters,
        return_hyper_values=False,
        run_steps_ahead_error="Other"):
    exact_gp = TrainTestPlotSaveExactGP(**input_parameters)
    bic_current_value, avg_er, er_lst, hyper_values, plotting_data, model_save = exact_gp \
        .run_train_test_plot_kernel(
            set_xlim=[0.996, 1],
            show_plots=True,
            return_np=True)
    if run_steps_ahead_error == "no_update":
        print("Running Steps Ahead Error...")
        err_list = exact_gp \
            .steps_ahead_error()
                # idx_list=index_list,
                # predict_ahead=predict_days_out)
        # plt.scatter(err_list)
        # plt.xlabel("Index")
        # plt.ylabel("Error")
        # plt.title(
        #     f'CV Random Split: Index vs Error '
        #     f'({input_parameters["kernel"]}: RMSE={np.nanmean(err_list):.3f})')
        # plt.show()
        print("Avg Error", np.nanmean(err_list))
        average_forecasting_error = np.nanmean(err_list)
    elif run_steps_ahead_error == "with_update":
        print("Running Steps Ahead Error... with updated hypers")
        metrics_updated = exact_gp \
            .step_ahead_update_model()
                # idx_list=index_list,
                # predict_ahead=predict_days_out)
        # print(metrics_updated)
        # plt.scatter(metrics_updated['err_list'][:])
        # plt.plot(metrics_updated['updated_bic_list'][:])
        # plt.show()
        # plt.title(f'OLD BIC: {bic_current_value} | NEW BIC: {metrics_updated["bic_list"][-1]}')
        average_forecasting_error = np.nanmean(metrics_updated['err_list'])
    else:
        average_forecasting_error = "Not Calculated"
    print("BIC", bic_current_value)
    print("Average Forecasting Error", avg_er)
    print("Hyper Values", hyper_values)
    # print("Error List", err_list)
    # temp_y_hat = exact_gp.test_y_hat.mean.detach().cpu().numpy()
    # temp_y = exact_gp.test_y.detach().cpu().numpy()
    del exact_gp
    gc.enable()
    gc.collect()
    torch.cuda.empty_cache()
    if return_hyper_values:
        return bic_current_value, avg_er, er_lst, hyper_values, plotting_data, model_save #, temp_y_hat, temp_y
    else:
        return bic_current_value, average_forecasting_error


def run_list_of_models(
        param_in_dict,
        list_of_models_to_try,
        file_name="model_results.csv",
        calculate_forecasting_error=True):
    model_results_output = []
    for try_model in list_of_models_to_try:
        param_in_dict["kernel"] = try_model
        param_in_dict["name"] = try_model
        b, av_e, e_l, h, plot_dat, model_di = run_the_model(
            param_in_dict,
            return_hyper_values=True,
            run_steps_ahead_error=calculate_forecasting_error)
        model_results_output.append([try_model, b, av_e, e_l, h, plot_dat, model_di])
        gc.enable()
        gc.collect()
        torch.cuda.empty_cache()
    model_results_df = pd.DataFrame(model_results_output)
    model_results_df.to_csv(file_name)
    return model_results_df

# def test_steps_ahead_error(
#         parameter_input_dictionary,
#         seeded_idx_list):
#     with_cv = []
#     past_trials = pd.read_csv(
#         "./../Past_Trials/full_results/cleaned_all_trials.csv")
#     past_trials.sort_values(by="BIC", inplace=True)
#     for index, row in past_trials[0:10].iterrows():
#         print(row["Kernel"])
#         parameter_input["kernel"] = str(row["Kernel"]).replace("'", "")
#         print("SAVED BIC: ", row["BIC"])
#         b, e = run_the_model(parameter_input_dictionary, seeded_idx_list)
#         with_cv.append([row["Kernel"], b, e])
#     with_cv_df = pd.DataFrame(with_cv)
#     with_cv_df.to_csv("top_10_trials_with_cv.csv")
#     print(with_cv_df)

# def find_index_size_stability(
#         checking_this_list_of_models=None,
#         start_size=100, end_size=2000, step_size=100):
#     if checking_this_list_of_models is None:
#         same_model_again_list = ["RBF+AR2+Mat_2.5", "RQ"]
#     for i in range(start_size, end_size, step_size):
#         same_model_again_list = list(checking_this_list_of_models)
#         same_model_results = run_list_of_models(
#             parameter_input,
#             make_idx_list(n_total, test_n, i),
#             same_model_again_list,
#             file_name=f'{i}_partition_size_model_list_{same_model_again_list[0]}.csv')
#         print(same_model_results)


# same_model_again_list = ["RBF+AR2+Mat_2.5", "RQ"]
# same_model_results = run_list_of_models(
#     parameter_input,
#     idx_list,
#     same_model_again_list,
#     file_name="same_model_results_2.csv")
# print(same_model_results)


# kl_list_temp = ["RBF+AR2*Per_Year*RBF*Mat_1.5*RBF", "RBF+AR2*Per_Year*RBF"]
trials_with_cv = pd.read_csv("top_10_trials_with_cv.csv")
list_of_kernels_to_try = trials_with_cv["0"].replace("'", "", regex=True)
kl_list_temp = [

    "RQ+AR2+Mat_2.5+Per_Year",
    "AR2+RBF",
    "RQ+AR2+Mat_2.5",
    "RBF+AR2*Per_Year*RBF",
    "AR2+RBF*AR2",
    # "AR2+RBF*Per_Year",

    "RQ+AR2+Mat_2.5+Per_Season*Mat_1.5",
    "RQ+AR2+Mat_2.5+Per_Season+Per_Season",
    # "RQ+AR2+Mat_2.5+Per_Season*Per_Season",
    # "RQ+AR2+Mat_2.5+Per_Season",
    # "AR2+RQ",
    # "AR2+Per_Year",
    # "AR2+Mat_2.5",

    # "RQ+AR2+Mat_2.5+Per_Season*Mat_0.5",
    # "RQ+AR2+Mat_2.5+Per_Season*Mat_2.5",
    # "RQ+AR2+Mat_2.5+Per_Season",
    # "RQ+AR2+Mat_2.5+Per_Month",
    # "RQ+AR2+Mat_2.5+Per_Week",
    # "RQ+AR2+Mat_2.5+Per_Season*Mat_1.5+Per_Year*RBF",
    # "RQ+AR2+Mat_2.5+Per_Season*Mat_1.5+Per_Month*RBF",
    # "RQ+AR2+Mat_2.5+Per_Season*Mat_1.5+Per_Month*Mat_2.5",
    # "RQ+AR2+Mat_2.5*Mat_2.5",

    # "RBF+AR2*Per_Year*RBF+Mat_1.5",
    # "RBF+AR2*Per_Year*RBF*Mat_1.5*RBF",
]
# kl_list_temp = kernel_list
print(kl_list_temp)
output_df_from_list = run_list_of_models(
    parameter_input,
    # kernel_list,
    # list_of_kernels_to_try,
    kl_list_temp,
    file_name="Run_top_list_fixed_plots_07_20_23_v1.csv",
    # calculate_forecasting_error='no_update'
)

print(output_df_from_list)
print(output_df_from_list.iloc[:, 0:3].sort_values(by=1, inplace=False).to_latex(index=False, float_format="{:.04f}".format,))
print(output_df_from_list.iloc[:, 0:3])


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
