import torch
import gpytorch
import tqdm.notebook
from gpytorch.constraints import Interval
from misc.custom_kernel import noise_lower, noise_upper, noise_init
from torch.utils.data import TensorDataset, DataLoader
import time
from matplotlib import pyplot as plt
import os
import sys
from pathlib import Path
# sys.path.append(Path(os.getcwd()).parent.__str__())


def get_BIC(model, likelihood, y, X_std):
    model.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).cuda()
    f = model(X_std)
    l = mll(f, y)  # log marginal likelihood
    num_param = sum(p.numel() for p in model.hyperparameters())
    BIC = -l * y.shape[0] + num_param / 2 * torch.tensor(y.shape[0]).log()
    return BIC


def set_gpytorch_settings():
    #gpytorch.settings.fast_computations.covar_root_decomposition._set_state(False)
    #gpytorch.settings.fast_computations.log_prob._set_state(False)
    #gpytorch.settings.fast_computations.solves._set_state(False)
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
    def __init__(
            self, model_cls, kernel,
            train_x, train_y, test_x, test_y,
            num_iter=50, debug=False, name=""):
        self.model_cls = model_cls
        self.kernel = kernel
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.num_iter = num_iter
        self.debug = debug
        self.name = str(name)

    def train_exact_gp(self):
        # start_time = time.time()
        likelihood = gpytorch \
            .likelihoods.GaussianLikelihood()
        model = self.model_cls(
            self.train_x, self.train_y, likelihood, self.kernel)
        # print(sys.path)
        smoke_test = ('CI' in os.environ)
        self.num_iter = 2 if smoke_test else 50
        torch.save(
            model.state_dict(),
            f'{str(self.name)}_{str(self.num_iter)}_iterations_PRE_train.pth')
        # Find optimal model hyper-parameters
        model.train()
        likelihood.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        if torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()
        num_iter_trys = tqdm.notebook.tqdm(
            range(self.num_iter), desc=f'Training_exactGP{self.name}')
        for i in num_iter_trys:
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()
        # torch.save(
        #     model.state_dict(),
        #     f'{int(self.num_iter)}_iterations_POST_train.pth')
        if self.debug:
            return model, likelihood, mll, optimizer, self.kernel
        else:
            return model, likelihood

    def test_eval_exact_gp(self):
        model_test, likelihood_test = self.train_exact_gp()
        model_test.eval()
        likelihood_test.eval()
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood_test(model_test(self.test_x))
            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.scatter(
            self.train_x[:, 0].detach().cpu().numpy(),
            self.train_y.detach().cpu().numpy(), s=0.5)
        # Plot predictive means as blue line
        ax.plot(
            self.test_x.detach().cpu().numpy(),
            observed_pred.mean.detach().cpu().numpy(),
            'blue')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(
            self.test_x[:, 0].detach().cpu().numpy(),
            lower.detach().cpu().numpy(),
            upper.detach().cpu().numpy(), alpha=0.3)
        ax.scatter(
            self.test_x[:, 0].detach().cpu().numpy(),
            self.test_y.detach().cpu().numpy(),
            s=1, color="red")
        ax.set_ylim([2.5, 3.5])
        ax.set_xlim(.8, 1)
        ax.legend(['Observed Data', 'Mean', 'Confidence', 'Predicted'])
        # ax.title("Exact GP")
        plt.savefig(f'{self.name}{str(self.num_iter)}_POST_test.png')
        plt.show()
        return model_test, likelihood_test, observed_pred, lower, upper
