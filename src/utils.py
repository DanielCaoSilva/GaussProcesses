import torch
import gpytorch
import tqdm.notebook
from gpytorch.constraints import Interval
from src.custom_kernel import noise_lower, noise_upper, noise_init
from torch.utils.data import TensorDataset, DataLoader
import time
from matplotlib import pyplot as plt
import os
import sys
from pathlib import Path
# sys.path.append(Path(os.getcwd()).parent.__str__())


def get_BIC(model, likelihood, y, X_std):
    with torch.no_grad():
        model.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).cuda()
        f = model(X_std)
        l = mll(f, y)  # log marginal likelihood
        num_param = sum(p.numel() for p in model.hyperparameters())
        BIC = -l * y.shape[0] + num_param / 2 * torch.tensor(y.shape[0]).log()
    return BIC


def set_gpytorch_settings(computations_state=False):
    gpytorch.settings.fast_computations.covar_root_decomposition._set_state(computations_state)
    gpytorch.settings.fast_computations.log_prob._set_state(computations_state)
    gpytorch.settings.fast_computations.solves._set_state(computations_state)
    gpytorch.settings.max_cholesky_size(100)
    #gpytorch.settings.debug._set_state(False)
    #gpytorch.settings.m
    #gpytorch.settings.min_fixed_noise._set_value(float_value=1e-7, double_value=1e-7, half_value=1e-7)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #torch.set_default_dtype(torch.float64)


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
            num_iter=100, debug=False, name="", lr=0.05,
            save_loss_values="save",
            use_scheduler=True,):
        self.use_scheduler = use_scheduler
        self.loss_values = []
        self.save_values = save_loss_values
        self.model_cls = model_cls
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
                optimizer, factor=0.50, patience=4, verbose=True)
        else:
            scheduler = None

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        # loocv = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, model)

        num_iter_trys = tqdm.notebook.tqdm(
            range(self.num_iter), desc=f'Training_exactGP{self.name}')
        for i in num_iter_trys:
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            # loss = -loocv(output, self.train_y)
            loss.backward()
            # if False:#self.save_values is not None:
            #     if self.save_values == "print":
            #         print(i + 1, loss.item())#,
            if self.save_values == "save":
                # print(
                #     i + 1,
                #     loss.item(),)
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
        f, ax = plt.subplots(1, 1, figsize=(10, 7))
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
        ax.legend(['Observed Data', 'Mean', 'Confidence', 'Predicted'])
        plt.title(f'Exact GP: {self.name}, {str(self.num_iter)}')
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

