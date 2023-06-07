import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.backends.cuda import *
import numpy as np
import pandas as pd
import gpytorch
import time
import tqdm.notebook
from gpytorch.constraints import Interval
from src.custom_kernel import noise_lower, noise_upper, noise_init
import time
import copy
from gpytorch.constraints import Interval, GreaterThan, LessThan
from matplotlib import pyplot as plt
from gpytorch.kernels import PeriodicKernel, ProductStructureKernel, AdditiveStructureKernel, ScaleKernel, \
    RBFKernel, MaternKernel, LinearKernel, PolynomialKernel, SpectralMixtureKernel, GridInterpolationKernel, \
    InducingPointKernel, ProductKernel, AdditiveKernel, GridKernel, RFFKernel, RQKernel
from gpytorch.means import ConstantMean
from src.custom_kernel import MinKernel, AR2Kernel
import os
import sys
from pathlib import Path
# sys.path.append(Path(os.getcwd()).parent.__str__())


# get_gpytorch_settings(): function to set gpytorch settings using in backend computation
def set_gpytorch_settings(computations_state=False):
    gpytorch.settings.fast_computations.covar_root_decomposition._set_state(computations_state)
    gpytorch.settings.fast_computations.log_prob._set_state(computations_state)
    gpytorch.settings.fast_computations.solves._set_state(computations_state)
    gpytorch.settings.max_cholesky_size(100)
    gpytorch.settings.cholesky_max_tries._set_value(50)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # gpytorch.settings.debug._set_state(False)
    # gpytorch.settings.m
    # gpytorch.settings.min_fixed_noise._set_value(float_value=1e-7, double_value=1e-7, half_value=1e-7)
    # torch.set_default_dtype(torch.float64)


# create_train_loader_and_dataset(): function to create train and test data loaders and datasets
def create_train_loader_and_dataset(train_x, train_y, test_x, test_y):
    train_dataset = TensorDataset(
        train_x, train_y)
    train_loader = DataLoader(
        train_dataset, batch_size=1024, shuffle=True,
        generator=torch.Generator(device='cuda'))
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(
        test_dataset, batch_size=1024, shuffle=False,
        generator=torch.Generator(device='cuda'))
    return train_loader, train_dataset, test_loader, test_dataset


# Returns the transformed hyperparameters from the raw ones obtained from the kernel
def get_named_parameters_and_constraints(kernel, print_out=False):
    hyper_values_transformed = []
    for iter_k in kernel.named_parameters_and_constraints():
        if print_out:
            print(iter_k[0], iter_k[2].transform(iter_k[1]).item())
        hyper_values_transformed.append([iter_k[0], iter_k[2].transform(iter_k[1]).item()])
    return hyper_values_transformed


def clean_csv_saves(
        csv_path="full_path_history_5_24_23.csv",
        out_path="full_path_history_5_24_23_cleaned.csv"):
    temp_path_loc = []
    temp_kernel_name = []
    temp_bic_value = []
    df = pd.read_csv(csv_path)
    new_df = df.iloc[:, 1:-1]
    unstacked_df = new_df.unstack()
    for i in unstacked_df:
        temp_split = i.split(", ")
        temp_path_loc.append(
            temp_split[0]
            .replace("[", "")+", "+temp_split[1])
        temp_kernel_name.append(temp_split[2])
        temp_bic_value.append(
            float(
                temp_split[3]
                .replace("]", "")))
    out_df = pd.DataFrame({
        "path_location": temp_path_loc,
        "Kernel": temp_kernel_name,
        "BIC": temp_bic_value})
    out_df.to_csv(csv_path.split(".")[0]+"_cleaned.csv", index=False)
    return out_df


# Kernel Utilities class: contains useful functions for the kernels
# scaler_min: minimum value of the scaler
# scaler_max: maximum value of the scaler
# scale_factor: scaler_max - scaler_min
# period_print: prints the period in different time units
# period_convert: converts the period from one time unit to another
# period_convert_list: converts a list of periods from one time unit to another
# generate_kernel_instance: generates a single kernel instance from a string
# make_kernel: generates a composite kernel from a string
class KernelUtils:
    def __init__(self, scaler_min, scaler_max):
        self.scaler_max = scaler_max
        self.scaler_min = scaler_min
        self.scale_factor = self.scaler_max - self.scaler_min

    def scaler(self, a, center=True):
        if center is True:
            a = a - self.scaler_min
        return a / self.scale_factor

    def period_print(self, x):
        print(f"raw: {x}")
        print(f"seconds: {x * self.scale_factor}")
        print(f"minutes: {x * self.scale_factor / 60}")
        print(f"hours: {x * self.scale_factor / 60 / 60}")
        print(f"days: {x * self.scale_factor / 60 / 60 / 24}")
        print(f"weeks: {x * self.scale_factor / 60 / 60 / 24 / 7}")
        print(f"months: {x * self.scale_factor / 60 / 60 / 24 / 30}")
        print(f"years: {x * self.scale_factor / 60 / 60 / 24 / 365}")

    def period_convert(self, x, type_to_convert):
        match type_to_convert:
            case "raw":
                return x
            case "seconds":
                return x * self.scale_factor
            case "minutes":
                return x * self.scale_factor / 60
            case "hours":
                return x * self.scale_factor / 60 / 60
            case "days":
                return x * self.scale_factor / 60 / 60 / 24
            case "weeks":
                return x * self.scale_factor / 60 / 60 / 24 / 7
            case "months":
                return x * self.scale_factor / 60 / 60 / 24 / 30
            case "years":
                return x * self.scale_factor / 60 / 60 / 24 / 365

    def period_convert_list(self, li, type_to_convert):
        converted_list = []
        for h in li:
            converted_list.append(self.period_convert(h, type_to_convert))
        return converted_list

    def generate_kernel_instance(self, full_kernel_string):
        kernel_class_parsed = str(full_kernel_string).split("_")
        kernel_class_type = kernel_class_parsed[0]
        if len(kernel_class_parsed) > 1:
            kernel_class_value = kernel_class_parsed[1]
        else:
            kernel_class_value = ""
        match str(kernel_class_type):
            case "RBF":
                return copy.deepcopy(RBFKernel(
                    lengthscale_constraint=GreaterThan(
                        0.000329)))
            case "RFF":
                return copy.deepcopy(RFFKernel(
                    num_samples=1024))  # 1024))
            case "Mat":
                return copy.deepcopy(MaternKernel(
                    nu=float(kernel_class_value),
                    lengthscale_constraint=GreaterThan(
                        0.000329)))
            case "AR2":
                return copy.deepcopy(AR2Kernel(
                    period_constraint=Interval(
                        lower_bound=1e-4, upper_bound=0.005),
                    lengthscale_constraint=GreaterThan(
                        0.000329)))
            case "Min":
                return copy.deepcopy(MinKernel())
            case "RQ":
                return copy.deepcopy(RQKernel(
                    alpha_constraint=GreaterThan(
                        0.000329),
                    lengthscale_constraint=GreaterThan(
                        0.000329)))
            case "Per":
                match kernel_class_value:
                    case "Unbounded":
                        return copy.deepcopy(PeriodicKernel())
                    case "Arb":
                        return copy.deepcopy(PeriodicKernel(
                            period_length_constraint=Interval(
                                lower_bound=1e-4, upper_bound=0.95),#0.005),
                            lengthscale_constraint=GreaterThan(
                                0.000329)))
                    case "Week":
                        return copy.deepcopy(PeriodicKernel(
                            period_length_constraint=Interval(
                                lower_bound=1e-4, upper_bound=0.5,
                                initial_value=self.scaler(60 * 60 * 24 * 7, center=False)),
                            lengthscale_constraint=GreaterThan(
                                0.000329)))
                    case "Month":
                        return copy.deepcopy(PeriodicKernel(
                            period_length_constraint=Interval(
                                lower_bound=1e-4, upper_bound=0.75,
                                initial_value=self.scaler(60 * 60 * 24 * 30, center=False)),
                            lengthscale_constraint=GreaterThan(
                                0.000329)))
                    case "Season":
                        return copy.deepcopy(PeriodicKernel(
                            period_length_constraint=Interval(
                                lower_bound=1e-4, upper_bound=0.75,
                                initial_value=self.scaler(60 * 60 * 24 * 120, center=False)),
                            lengthscale_constraint=GreaterThan(
                                0.000329)))
                    case "Year":
                        return copy.deepcopy(PeriodicKernel(
                            period_length_constraint=Interval(
                                lower_bound=1e-4, upper_bound=0.8,
                                initial_value=self.scaler(60 * 60 * 24 * 365, center=False)),
                            lengthscale_constraint=GreaterThan(
                                0.000329)))

    def make_kernel(self, name_of_kernel):
        return_kernel_list = []
        kernel_additive_terms = str(name_of_kernel).split('+')
        for add_term_index, add_term in enumerate(kernel_additive_terms):
            kernel_mult_terms = str(add_term).split("*")
            for mult_term_index, mult_term in enumerate(kernel_mult_terms):
                return_kernel_list.append(mult_term)
                if mult_term_index == 0:
                    cum_prod = self.generate_kernel_instance(mult_term)
                else:
                    cum_prod = cum_prod * self.generate_kernel_instance(mult_term)
            if add_term_index == 0:
                cum_sum = copy.deepcopy(ScaleKernel(cum_prod))
            else:
                cum_sum = cum_sum + copy.deepcopy(ScaleKernel(cum_prod))
        return copy.deepcopy(cum_sum)  # , return_kernel_list


# TrainTestPlotSaveExactGP class: train, test, plot, save
# train_exact_gp: trains the model and saves the trained model and likelihood
# test_exact_gp: tests the model and saves the test predictions
# plot_exact_gp: plots the test predictions
# get_bic: returns the BIC of the model
# run_train_test_plot: runs the train, test, plot, save functions
# run_train_test_plot_kernel: runs the train, test, plot, save functions and returns the hyperparameters
class TrainTestPlotSaveExactGP:
    test_y_hat = None
    lower, upper = None, None
    trained_model = None
    trained_likelihood = None
    eval_model = None
    eval_likelihood = None
    BIC = None
    status_check = {
        "train": False,
        "test": False,
        "plot": False
    }

    def __init__(
            self, model_cls, kernel,
            train_x, train_y, test_x, test_y,
            scaler_min, scaler_max,
            num_iter=100, debug=False, name="", lr=0.01,
            save_loss_values="save",
            use_scheduler=True,):
        self.ku = KernelUtils(scaler_min, scaler_max)
        self.use_scheduler = use_scheduler
        self.loss_values = []
        self.save_values = save_loss_values
        self.model_cls = model_cls
        if type(kernel) == str:
            self.kernel = copy.deepcopy(self.ku.make_kernel(kernel))
        else:
            self.kernel = kernel
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.num_iter = num_iter
        self.debug = debug
        self.name = str(name)
        self.lr = lr

    def train_exact_gp(self):
        # start_time = time.time()
        # if self.status_check["train"] is True:
        #     return True
        likelihood = gpytorch \
            .likelihoods.GaussianLikelihood()
        model = self.model_cls(
            self.train_x, self.train_y, likelihood, self.kernel)

        if torch.cuda.is_available():
            print("Using available CUDA")
            # self.train_x = self.train_x.cuda()
            # self.train_y = self.train_y.cuda()
            model = model.cuda()
            likelihood = likelihood.cuda()
        else:
            print("CUDA is not active")

        # Find optimal model hyper-parameters
        model.train()
        likelihood.train()
        # Use the adam optimizer
        # Learning Rate: (Alpha), or the step size, is the ratio of parameter update to
        # gradient/momentum/velocity depending on the optimization algo
        # Typically staying with ing 0.0001 and 0.01
        # Betas:
        # beta1 is the exponential decay rate for the momentum term also called first moment estimate
        # beta2 is the exponential decay rate for the velocity term also called the second-moment estimates.
        # Weight Decay: [default: 0] avoids overshooting the minima often resulting in faster convergence of the
        # loss function
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, ## https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
            weight_decay=1e-8, betas=(0.9, 0.999), eps=1e-7)  # Includes GaussianLikelihood parameters

        # Scheduler - Reduces alpha by [factor] every [patience] epoch that does not improve based on
        # loss input
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.40, patience=4, verbose=False, eps=1e-7)
        else:
            scheduler = None

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        # loocv = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, model)

        # num_iter_trys = tqdm.notebook.\
        num_iter_trys = tqdm.tqdm(
            range(self.num_iter), desc=f'Training exactGP: {self.name}')
        for i in num_iter_trys:
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            # loss = -loocv(output, self.train_y)
            # loss_candidate = -mll(output, self.train_y)
            # if loss_candidate is torch.nan:
            #     for g in optimizer.param_groups:
            #         g['lr'] = g['lr']/2
            # else:
            #     loss = loss_candidate

            loss.backward()
            # if False:#self.save_values is not None:
            #     if self.save_values == "print":
            #         print(i + 1, loss.item())#,
            if self.save_values == "save":
                # print(i + 1, loss.item(),)
                    # model.covar_module.base_kernel.lengthscale.item(),)
                    # model.covar_module.kernels[0].base_kernel.lengthscale.item()) #, model.covar_module.kernels)#.kernels[0].lengthscale.item())
                self.loss_values.append([i+1, loss.item()])#, model.covar_module.base_kernel.lengthscale.item()])
            optimizer.step()
            if self.use_scheduler:
                scheduler.step(loss)
        self.status_check["train"] = True
        self.trained_model = model
        self.trained_likelihood = likelihood
        # print(model.state_dict())
        if self.debug:
            return self.model, self.likelihood, mll, optimizer, self.kernel
        # else:
        #     return self.model, self.likelihood

    def test_eval_exact_gp(self):
        # if self.status_check["train"] is False:
        #     self.train_exact_gp()
        self.train_exact_gp()
        # Set to eval mode
        self.eval_model = self.trained_model.eval()
        self.eval_likelihood = self.trained_likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # self.test_x = self.test_x
            self.test_y_hat = self.eval_likelihood(self.eval_model(self.test_x))
            # Get upper and lower confidence bounds
            self.lower, self.upper = self.test_y_hat.confidence_region()
        # error = torch.mean(torch.abs(self.test_y_hat.mean - self.test_y))
        self.status_check["test"] = True
        self.trained_model.train()
        self.trained_likelihood.train()
        # return self.model, self.likelihood, self.test_y_hat, self.lower, self.upper, error

    def plot(self, set_x_limit=(0, 1), set_y_limit=None, show_plot=True):
        # if self.status_check["test"] is False:
        #     self.test_eval_exact_gp()
        f, (ax, axLoss) = plt.subplots(2, 1, figsize=(10, 7))
        plt.title(f'Exact GP: {self.name}, Trials: {str(self.num_iter)}, BIC: {self.get_BIC().item()}')
        # Plot training data as black stars
        ax.scatter(
            self.train_x[:, 0].detach().cpu().numpy(),
            self.train_y.detach().cpu().numpy(), s=0.5)
        # Plot predictive means as blue line
        ax.plot(
            self.test_x.detach().cpu().numpy(),
            self.test_y_hat.mean.detach().cpu().numpy(),
            'blue')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(
            self.test_x[:, 0].detach().cpu().numpy(),
            self.lower.detach().cpu().numpy(),
            self.upper.detach().cpu().numpy(), alpha=0.3)
        ax.scatter(
            self.test_x[:, 0].detach().cpu().numpy(),
            self.test_y.detach().cpu().numpy(),
            s=1, color="red")
        if set_x_limit is not None:
            ax.set_xlim([set_x_limit[0], set_x_limit[1]])
        if set_y_limit is not None:
            ax.set_ylim([set_y_limit[0], set_y_limit[1]])
        ax.legend(['Observed Data', 'Mean', 'Confidence', 'Predicted'], fontsize='x-small', loc='upper left')
        pd.DataFrame(self.loss_values).plot(x=0, y=1, ax=axLoss)
        # axLoss.scatter(self.loss_values, s=0.5)
        # axLoss.title("Iterations vs Loss")
        # plt.savefig(f'Trials\\{str(self.name).replace(".", "")}{str(self.num_iter)}_POST_test.png')
        if show_plot:
            plt.show()

    def get_BIC(self):
        with torch.no_grad():
            self.trained_model.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.trained_likelihood, self.trained_model).cuda()
            f = self.trained_model(self.train_x)
            l = mll(f, self.train_y)  # log marginal likelihood
            num_param = sum(p.numel() for p in self.trained_model.hyperparameters())
            self.BIC = -l * self.train_y.shape[0] + num_param / 2 * torch.tensor(self.train_y.shape[0]).log()
        return self.BIC

    def run_train_test_plot(self, set_xlim=None):
        self.test_eval_exact_gp()
        self.plot(show_plot=True, set_x_limit=set_xlim)

    def run_train_test_plot_kernel(self, set_xlim=None, show_plots=False):
        self.test_eval_exact_gp()
        self.plot(show_plot=show_plots, set_x_limit=set_xlim)
        bic_calc = self.get_BIC().item()
        hyper_params = get_named_parameters_and_constraints(self.kernel, print_out=False)
        return bic_calc, hyper_params
    # def close(self):jj
    #     self.model_cls = None
    #     self.trainedmodel = None
    #     self.likelihood = None
    #     self.test_y_hat = None
    #     self.lower = None
    #     self.upper = None
    #     self.BIC = None
    #     self.status_check = {
    #         "train": False,
    #         "test": False,NM,J,NM,..,,.,.KJ.MN,.M,.......
    #         "plot": False
    #     }
    #     self.train_x = None
    #     self.train_y = None
    #     self.test_x = None
    #     self.test_y = None
    #     self.num_iter = Noneddddddddddddddddddddwwwwwwwwwwwwwwwwwww1111111111111111111111111bbbbbbbbbbbbbbbbb

    def steps_ahead_error(self, idx_list, predict_ahead):
        start_time = time.time()
        test_n_points_ahead = predict_ahead * 2
        err_list = []
        for idx_ahead in idx_list:
            # print(iter)
            with torch.no_grad():
                # picks index in the second half of the observed data
                # print(idx_ahead)
                # going to be the new "training" set; from t=0 to t=idx_ahead-1
                temp_x_train = self.train_x[:idx_ahead]
                temp_y_train = self.train_y[:idx_ahead]
                # this is the new "testing" set: from t=idx_ahead, ..., t=idx_ahead+6
                temp_x_test = self.train_x[idx_ahead:(idx_ahead+test_n_points_ahead)]
                temp_y_test = self.train_y[idx_ahead:(idx_ahead+test_n_points_ahead)]
                self.trained_model.set_train_data(
                    inputs=temp_x_train, targets=temp_y_train, strict=False)
                self.trained_model.eval()
                # grab the evaluated model predictions
                f = self.trained_model(temp_x_test)
                # calculate the error (MSE) for the prediction
                err = torch.mean((f.mean - temp_y_test)**2).item()
                err_list.append(err)
        print("--- %s seconds ---" % (time.time() - start_time))
        return err_list


# get_BIC: function to calculate BIC given model, likelihood, data
def get_BIC(model, likelihood, y, X_std):
    with torch.no_grad():
        model.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).cuda()
        f = model(X_std)
        l = mll(f, y)  # log marginal likelihood
        num_param = sum(p.numel() for p in model.hyperparameters())
        BIC = -l * y.shape[0] + num_param / 2 * torch.tensor(y.shape[0]).log()
    return BIC


# train_and_test_exact_gp: function to train and test approximate GP
def train_and_test_approximate_gp(
        model_cls, kernel,
        train_x, train_y, test_x, test_y,
        train_loader, train_dataset, test_loader, test_dataset,
        num_ind_pts=128, num_epochs=100):
    start_time = time.time()
    inducing_points = torch.randn(
        num_ind_pts, train_x.size(-1),
        dtype=train_x.dtype, device=train_x.device)
    model = model_cls(inducing_points, kernel)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=Interval(noise_lower, noise_upper, initial_value=noise_init))
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.numel())
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=0.1)
    optimizer.param_groups[0]['capturable'] = True

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    # Training
    model.train()
    likelihood.train()
    epochs_iter = tqdm.notebook.tqdm(
        range(num_epochs), desc=f"Training {model_cls.__name__}")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            # print(loss, loss.shape)
            epochs_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

    # Testing
    model.eval()
    likelihood.eval()
    means = torch.tensor([0.]).cuda()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            means = torch.cat([means, preds.mean])
    means = means[1:]
    error = torch.mean(torch.abs(means - test_y))
    print(f"Test {model_cls.__name__} MAE: {error.item()}")
    print("--- %s seconds ---" % (time.time() - start_time))
    return model, likelihood




