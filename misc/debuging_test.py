#%%
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from misc import utils
from skimage.measure import block_reduce


plt.style.use('classic')

from misc.utils import set_gpytorch_settings

set_gpytorch_settings()

# Command in terminal to help with memory allocation
# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
#%%
# Kernel Imports
from gpytorch.kernels import PeriodicKernel
from misc.custom_kernel import MaternKernel
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.constraints import Interval
from misc.custom_kernel import noise_lower, noise_upper, noise_init
#%%
df = pd.read_feather('../data/feather/46221_9999_wave_height.feather')
df_as_np = df \
    .loc[:, ['time', 'sea_surface_temperature']] \
    .astype(float) \
    .replace(to_replace = [999.0, 99.0, 9999.0], value = np.nan) \
    .to_numpy()
print(len(df_as_np))
print(df_as_np)
using_sk = block_reduce(df_as_np, block_size=(24,1), func=np.mean).astype(float)
print(len(using_sk))
print(using_sk)
plt.plot(using_sk[:-1,0], using_sk[:-1,1])

#%%
(print(using_sk[:-1,1]))
#%%
X = torch.tensor(using_sk[:-1,0]).float().cuda()#.type(torch.double)
y = torch.tensor(using_sk[:-1,1]).float().cuda()#.type(torch.double)
X = X.reshape(-1,1)
y = y.reshape(-1,1)
X = X[~torch.any(y.isnan(),dim=1)]
y = y[~torch.any(y.isnan(),dim=1)]
X_old = X
# print(X,y)
print(y)#
print(X)
#%%
print(y.shape)
print(X.shape)
print(y.sum())
#%%
def scaler(a, X_old=X_old, center=True):
    if center is True:
        a = a - X_old.min(0).values
    return a / (X_old.max(0).values - X_old.min(0).values)
X = scaler(X, X_old)

print(y.sum())
y = y.log()
# y = y - torch.min(y)
# y = 2 * (y / torch.max(y)) - 1
print(y.sum())
#%%
print(len(X), len(y))
#%%
test_n = 800
train_x = X[test_n:].contiguous().cuda()
train_y = y[test_n:].contiguous().cuda()
test_x = X[-test_n:].contiguous().cuda()
test_y = y[-test_n:].contiguous().cuda()

print(train_x.min())
print(train_x.max())
print(train_x.mean())

print(test_x.min())
print(test_x.max())
print(test_x.mean())
#%%
# Generate the train_loader and train_dataset
train_loader, train_dataset, test_loader, test_dataset = utils.create_train_loader_and_dataset(
    train_x, train_y, test_x, test_y)
data_compact = [train_x, train_y, test_x, test_y, train_loader, train_dataset, test_loader, test_dataset]
#%%
train_y.sum()
#%%
class StandardApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(-2))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MeanFieldApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel):
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(-2))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MAPApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel):
        variational_distribution = gpytorch.variational.DeltaVariationalDistribution(inducing_points.size(-2))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def make_orthogonal_vs(model, train_x):
    mean_inducing_points = torch.randn(1000, train_x.size(-1), dtype=train_x.dtype, device=train_x.device)
    covar_inducing_points = torch.randn(100, train_x.size(-1), dtype=train_x.dtype, device=train_x.device)

    covar_variational_strategy = gpytorch.variational.VariationalStrategy(
        model, covar_inducing_points,
        gpytorch.variational.CholeskyVariationalDistribution(covar_inducing_points.size(-2)),
        learn_inducing_locations=True
    )

    variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
        covar_variational_strategy, mean_inducing_points,
        gpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(-2)),
    )
    return variational_strategy

class OrthDecoupledApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel):
        variational_distribution = gpytorch.variational.DeltaVariationalDistribution(inducing_points.size(-2))
        variational_strategy = make_orthogonal_vs(self, train_x)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SpectralDeltaGP(gpytorch.models.ExactGP):
    # def __init__(self, train_x, train_y, kernel, num_deltas, noise_init=None):
    #     likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-11))
    #     likelihood.register_prior("noise_prior", gpytorch.priors.HorseshoePrior(0.1), "noise")
    #     likelihood.noise = 1e-2
    def __init__(self, inducing_points, kernel):
        variational_distribution = gpytorch.variational.DeltaVariationalDistribution(inducing_points)
        #variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(-2))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        #super(SpectralDeltaGP, self).__init__(train_x, train_y, likelihood)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        #base_covar_module = kernel #gpytorch.kernels.SpectralDeltaKernel(num_dims=train_x.size(-1), num_deltas=num_deltas)
        #base_covar_module.initialize_from_data(train_x[0], train_y[0])
        self.covar_module = kernel#gpytorch.kernels.ScaleKernel(base_covar_module)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



#likelihood = gpytorch.likelihoods.GaussianLikelihood()
#model = SpectralMixtureGPModel(train_x, train_y, likelihood)
#%%
Mat32 = MaternKernel(nu=1.5)
Mat12 = MaternKernel(nu=0.5)
RBF = RBFKernel()

# Per_Day = PeriodicKernel(
#     period_length_constraint=Interval(
#     lower_bound=scaler(60*24, center=False) / 100,
#     upper_bound=scaler(60*24, center=False) * 100,
#     initial_value=scaler(60*24, center=False))
# )

# Per_Month = PeriodicKernel(
#     period_length_constraint=Interval(
#     lower_bound=scaler(60*24*30, center=False) / 100,
#     upper_bound=scaler(60*24*30, center=False) * 100,
#     initial_value=scaler(60*24*30, center=False))
# )
Per_Arb = PeriodicKernel()
# kernel = (
# 	k1 + k3 + k4
#     # not: ScaleKernel(k1+k2) since they all have spectral component
# )

kernel = (
    ScaleKernel(Mat12) +
	ScaleKernel(RBF) + ScaleKernel(Per_Arb)
    # ScaleKernel(Per_Day) +
    # ScaleKernel(Per_Month)
    # not: ScaleKernel(k1+k2) since they all have spectral component
)


likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint = Interval(noise_lower, noise_upper,initial_value=noise_init))
#%%
y
#%%
# gpytorch.settings.cholesky_max_tries._set_value(100)
# gpytorch.settings.max_cholesky_size._set_value(1000)
# gpytorch.settings.cholesky_jitter._set_value(float_value=1e-2, double_value = 1e-2, half_value = 1e-2)
gpytorch.settings.cholesky_jitter._set_value(double_value=1e8, float_value=1e-4, half_value=1e-3)
num_ind_pts = 128 # Number of inducing points (128 is default for train_and_test_approximate_gp function)
num_epochs = 20

#m1, l1 = utils.train_and_test_approximate_gp(
#    StandardApproximateGP, kernel, *data_compact, num_epochs=100, num_ind_pts=num_ind_pts)
# m2, l2 = utils.train_and_test_approximate_gp(
#     MeanFieldApproximateGP, kernel, *data_compact, num_epochs=100, num_ind_pts=num_ind_pts)
# m3, l3 = utils.train_and_test_approximate_gp(
#     MAPApproximateGP, kernel, *data_compact, num_epochs=100, num_ind_pts=num_ind_pts)
# m4, l4 = utils.train_and_test_approximate_gp(
#     OrthDecoupledApproximateGP, kernel, *data_compact, num_epochs=100, num_ind_pts=num_ind_pts)
# m5, l5 = utils.train_and_test_approximate_gp(
#     SpectralDeltaGP, kernel, *data_compact, num_epochs=100, num_ind_pts=num_deltas)
# m6, l6 = utils.train_and_test_approximate_gp(
#     SpectralDeltaGP, kernel, *data_compact, num_epochs=100, num_ind_pts=num_ind_pts)
#l1 = gpytorch.likelihoods.GaussianLikelihood()
#m1 = SpectralMixtureGPModel(train_x, train_y, likelihood)
m1, l1 = utils.train_and_test_approximate_gp(
    StandardApproximateGP, kernel, *data_compact, num_epochs=num_epochs, num_ind_pts=num_ind_pts)
#print(kernel.kernels[2].base_kernel.lengthscale)
m2, l2 = utils.train_and_test_approximate_gp(
    OrthDecoupledApproximateGP, kernel, *data_compact, num_epochs=num_epochs, num_ind_pts=num_ind_pts)
#print(kernel.kernels[2].base_kernel.lengthscale)
#%%
pairs = [[m1, l1],
        [m2, l2],]# [m3, l3],
#         [m4, l4], [m5, l5],
        #[m1, l1]]

for pair in pairs:
    model = pair[0]
    likelihood = pair[1]
    model.eval()
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(10, 7))
    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x[:,0].detach().cpu().numpy(),
                    lower.detach().cpu().numpy(),
                    upper.detach().cpu().numpy(), alpha=0.3)
    # Plot training data as black stars
    ax.scatter(train_x[:,0].detach().cpu().numpy(), train_y.detach().cpu().numpy(), s=0.5)
    #ax.scatter(model.variational_strategy.inducing_points[:,0].detach().cpu().numpy(),
    #           np.zeros(500)+1, s=0.5)
    # Plot predictive means as blue line
    ax.plot(test_x[:,0].detach().cpu().numpy(), observed_pred.mean.detach().cpu().numpy(), 'blue')
    ax.scatter(
        test_x[:,0].detach().cpu().numpy(),
        test_y.detach().cpu().numpy(),
        s=1, color="red")
    # ax.set_xlim(.999,1)
    ax.set_xlim(0,1)
    #ax.set_xlim(0.65,0.75)
    ax.vlines(m1.variational_strategy.inducing_points.detach().cpu().numpy(), ymin = -1.6, ymax = -1.5)

    #ax.set_ylim([0, 1.5])
    #ax.patch.set_facecolor('green')
    #ax.patch.set_alpha(.1)
    ax.legend(["95% Credible Intervals", "Observed Data", "Posterior Mean"])