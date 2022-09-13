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