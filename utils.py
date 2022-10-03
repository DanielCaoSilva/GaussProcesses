import torch
import gpytorch


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
    gpytorch.settings.cholesky_max_tries._set_value(100)
    #gpytorch.settings.debug._set_state(False)
    gpytorch.settings.min_fixed_noise._set_value(float_value=1e-7, double_value=1e-7, half_value=1e-7)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #torch.set_default_dtype(torch.float64)