#!/usr/bin/env python3

import warnings

import torch

from gpytorch.constraints import Interval, Positive
from gpytorch.lazy import MatmulLazyTensor, RootLazyTensor, LazyEvaluatedKernelTensor, ZeroLazyTensor, delazify, lazify
from gpytorch.priors import Prior
from gpytorch.kernels.kernel import Kernel, AdditiveKernel, ProductKernel
from gpytorch.module import Module
from gpytorch import settings
from gpytorch.functions import RBFCovariance
from gpytorch.settings import trace_mode
from gpytorch.functions import MaternCovariance
from torch.nn import ModuleList
from copy import deepcopy
import math
from typing import Optional, Tuple
from abc import abstractmethod
#from gpytorch.utils.broadcasting import _mul_broadcast_shape
from gpytorch.models import exact_prediction_strategies
#from utils import set_gpytorch_settings

#set_gpytorch_settings()


c_lower = torch.tensor(0.001)
c_upper = torch.tensor(1000)
c_init = torch.tensor(0.04)

offset_lower = torch.tensor(0.0001)
offset_upper = torch.tensor(1000)
offset_init = torch.tensor(0.1)


l_rbf_lower = torch.tensor(0.05)
l_rbf_upper = torch.tensor(5)
l_rbf_init = torch.tensor(0.20)

l_m12_lower = torch.tensor(0.005)
l_m12_upper = torch.tensor(5)
l_m12_init = torch.tensor(0.4)

l_m32_lower = torch.tensor(0.01)
l_m32_upper = torch.tensor(2.5)
l_m32_init = torch.tensor(0.25)

l_m52_lower = torch.tensor(0.025)
l_m52_upper = torch.tensor(7.5)
l_m52_init = torch.tensor(0.225)

l_chy_lower = torch.tensor(1e-5)
l_chy_upper = torch.tensor(7)
l_chy_init = torch.tensor(0.2)

noise_lower = torch.tensor(1e-5)
noise_upper = torch.tensor(1.0)
noise_init = torch.tensor(1e-2)

per_lower = torch.tensor(0.1)
per_upper = torch.tensor(1.0)
per_init = torch.tensor(0.5)


class MinKernel(Kernel):

    def __init__(
        self,
        offset_prior: Optional[Prior] = None,
        offset_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        super(MinKernel, self).__init__(**kwargs)
        if offset_constraint is None:
            offset_constraint = Interval(offset_lower, offset_upper, initial_value=offset_init)
        self.register_parameter(name="raw_offset", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))

        if offset_prior is not None:
            if not isinstance(offset_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(offset_prior).__name__)
            self.register_prior("offset_prior", offset_prior, lambda m: m.offset, lambda m, v: m._set_offset(v))

        self.register_constraint("raw_offset", offset_constraint)

    @property
    def offset(self) -> torch.Tensor:
        return self.raw_offset_constraint.transform(self.raw_offset)

    @offset.setter
    def offset(self, value: torch.Tensor) -> None:
        self._set_offset(value)

    def _set_offset(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_offset)
        self.initialize(raw_offset=self.raw_offset_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1_ = x1
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
        a = torch.ones(x1_.shape)
        # a.to(x1_.device)  # for cuda vs cpu  (I don't think this works - may need fixing later)
        offset = self.offset.view(*self.batch_shape, 1, 1)

        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLazyTensor when x1 == x2 for efficiency when composing
            # with other kernels
            aa = MatmulLazyTensor(x1_, a.transpose(-2, -1))
            bb = aa.transpose(-2,-1)
            K = 0.5 * (aa + bb - torch.abs((aa - bb).evaluate())) + offset

        else:
            x2_ = x2
            if last_dim_is_batch:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)
            b = torch.ones(x2_.shape)
            # b.to(x2_.device)  # for cuda vs cpu  (I don't think this works - may need fixing later)
            aa = MatmulLazyTensor(x1_, b.transpose(-2, -1))
            bb = MatmulLazyTensor(a, x2_.transpose(-2, -1))
            K = 0.5 * (aa + bb - torch.abs((aa - bb).evaluate())) + offset
        if diag:
            return K.diag()
        else:
            return K


class AR2Kernel(Kernel):
    # Berlinet et. al p317

    has_lengthscale = False

    def __init__(
        self,
        lengthscale_prior: Optional[Prior] = None,
        lengthscale_constraint: Optional[Interval] = None,
        period_prior: Optional[Prior] = None,
        period_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        super(AR2Kernel, self).__init__(**kwargs)
        if lengthscale_constraint is None:
            lengthscale_constraint = Interval(
                torch.tensor(0.05),
                torch.tensor(500),
                initial_value=torch.tensor(10))
        if period_constraint is None:
            period_constraint = Interval(1e-4, 5, initial_value=0.75)

        self.register_parameter(
            name="raw_lengthscale", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_parameter(
            name="raw_period", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        if lengthscale_prior is not None:
            if not isinstance(lengthscale_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(lengthscale_prior).__name__)
            self.register_prior(
                "lengthscale_prior",
                lengthscale_prior,
                lambda m: m.lengthscale,
                lambda m, v: m._set_lengthscale(v),
            )
        if period_prior is not None:
            if not isinstance(period_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(period_prior).__name__)
            self.register_prior(
                "period_prior",
                period_prior,
                lambda m: m.period,
                lambda m, v: m._set_period(v),
            )

        self.register_constraint("raw_lengthscale", lengthscale_constraint)
        self.register_constraint("raw_period", period_constraint)

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @property
    def period(self):
        return self.raw_period_constraint.transform(self.raw_period)

    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)
    def _set_lengthscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))

    @period.setter
    def period(self, value):
        self._set_period(value)

    def _set_period(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period)
        self.initialize(raw_period=self.raw_period_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        diff = self.covar_dist(x1, x2, diag=diag, **params)
        pi = torch.tensor(torch.pi)
        #gamma_sq = self.period.pow(2) + self.lengthscale.pow(2)
        a = torch.exp(-diff.div(self.lengthscale))
        b = torch.cos(diff.mul(pi).div(self.period))
        c = torch.sin(diff.mul(pi).div(self.period)).mul(self.period).div(self.lengthscale).div(pi)
        # const = self.lengthscale.div(4).div(torch.add(
        #     torch.tensor(1).div(self.lengthscale.pow(2)),
        #     pi.pow(2).div(self.period.pow(2))))

        #res = a.mul(b).mul(const) + c
        res = (b+c).mul(a)
        if diag:
            res = res.squeeze(0)
        return res

#
# def default_postprocess_script(x):
#     return x
#
# class Distance(torch.nn.Module):
#     def __init__(self, postprocess_script=default_postprocess_script):
#         super().__init__()
#         self._postprocess = postprocess_script
#
#     def _sq_dist(self, x1, x2, postprocess, x1_eq_x2=False):
#         # TODO: use torch squared cdist once implemented: https://github.com/pytorch/pytorch/pull/25799
#         adjustment = x1.mean(-2, keepdim=True)
#         x1 = x1 - adjustment
#         x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point
#
#         # Compute squared distance matrix using quadratic expansion
#         x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
#         x1_pad = torch.ones_like(x1_norm)
#         if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
#             x2_norm, x2_pad = x1_norm, x1_pad
#         else:
#             x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
#             x2_pad = torch.ones_like(x2_norm)
#         x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
#         x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
#         res = x1_.matmul(x2_.transpose(-2, -1))
#
#         if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
#             res.diagonal(dim1=-2, dim2=-1).fill_(0)
#
#         # Zero out negative values
#         res.clamp_min_(0)
#         return self._postprocess(res) if postprocess else res
#
#     def _dist(self, x1, x2, postprocess, x1_eq_x2=False):
#         # TODO: use torch cdist once implementation is improved: https://github.com/pytorch/pytorch/pull/25799
#         res = self._sq_dist(x1, x2, postprocess=False, x1_eq_x2=x1_eq_x2)
#         res = res.clamp_min_(1e-30).sqrt_()
#         return self._postprocess(res) if postprocess else res
#
#
# class Kernel(Module):
#     has_lengthscale = False
#
#     def __init__(
#         self,
#         ard_num_dims: Optional[int] = None,
#         batch_shape: Optional[torch.Size] = torch.Size([]),
#         active_dims: Optional[Tuple[int, ...]] = None,
#         lengthscale_prior: Optional[Prior] = None,
#         lengthscale_constraint: Optional[Interval] = None,
#         eps: Optional[float] = 1e-6,
#         **kwargs,
#     ):
#         super(Kernel, self).__init__()
#         self._batch_shape = batch_shape
#         if active_dims is not None and not torch.is_tensor(active_dims):
#             active_dims = torch.tensor(active_dims, dtype=torch.long)
#         self.register_buffer("active_dims", active_dims)
#         self.ard_num_dims = ard_num_dims
#
#         self.eps = eps
#
#         param_transform = kwargs.get("param_transform")
#
#         if lengthscale_constraint is None:
#             lengthscale_constraint = Interval(l_lower, l_upper, initial_value=l_init)
#
#         if param_transform is not None:
#             warnings.warn(
#                 "The 'param_transform' argument is now deprecated. If you want to use a different "
#                 "transformation, specify a different 'lengthscale_constraint' instead.",
#                 DeprecationWarning,
#             )
#
#         if self.has_lengthscale:
#             lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
#             self.register_parameter(
#                 name="raw_lengthscale",
#                 parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, lengthscale_num_dims)),
#             )
#             if lengthscale_prior is not None:
#                 if not isinstance(lengthscale_prior, Prior):
#                     raise TypeError("Expected gpytorch.priors.Prior but got " + type(lengthscale_prior).__name__)
#                 self.register_prior(
#                     "lengthscale_prior", lengthscale_prior, self._lengthscale_param, self._lengthscale_closure
#                 )
#
#             self.register_constraint("raw_lengthscale", lengthscale_constraint)
#
#         self.distance_module = None
#         # TODO: Remove this on next official PyTorch release.
#         self.__pdist_supports_batch = True
#
#     @abstractmethod
#     def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
#         raise NotImplementedError()
#
#     @property
#     def batch_shape(self):
#         kernels = list(self.sub_kernels())
#         if len(kernels):
#             return _mul_broadcast_shape(self._batch_shape, *[k.batch_shape for k in kernels])
#         else:
#             return self._batch_shape
#
#     @batch_shape.setter
#     def batch_shape(self, val):
#         self._batch_shape = val
#
#     @property
#     def dtype(self):
#         if self.has_lengthscale:
#             return self.lengthscale.dtype
#         else:
#             for param in self.parameters():
#                 return param.dtype
#             return torch.get_default_dtype()
#
#     @property
#     def is_stationary(self) -> bool:
#         """
#         Property to indicate whether kernel is stationary or not.
#         """
#         return self.has_lengthscale
#
#     def _lengthscale_param(self, m):
#         return m.lengthscale
#
#     def _lengthscale_closure(self, m, v):
#         return m._set_lengthscale(v)
#
#     @property
#     def lengthscale(self):
#         if self.has_lengthscale:
#             return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
#         else:
#             return None
#
#     @lengthscale.setter
#     def lengthscale(self, value):
#         self._set_lengthscale(value)
#
#     def _set_lengthscale(self, value):
#         if not self.has_lengthscale:
#             raise RuntimeError("Kernel has no lengthscale.")
#
#         if not torch.is_tensor(value):
#             value = torch.as_tensor(value).to(self.raw_lengthscale)
#
#         self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))
#
#     def local_load_samples(self, samples_dict, memo, prefix):
#         num_samples = next(iter(samples_dict.values())).size(0)
#         self.batch_shape = torch.Size([num_samples]) + self.batch_shape
#         super().local_load_samples(samples_dict, memo, prefix)
#
#     def covar_dist(
#         self,
#         x1,
#         x2,
#         diag=False,
#         last_dim_is_batch=False,
#         square_dist=False,
#         dist_postprocess_func=default_postprocess_script,
#         postprocess=True,
#         **params,
#     ):
#         r"""
#         This is a helper method for computing the Euclidean distance between
#         all pairs of points in x1 and x2.
#         Args:
#             :attr:`x1` (Tensor `n x d` or `b1 x ... x bk x n x d`):
#                 First set of data.
#             :attr:`x2` (Tensor `m x d` or `b1 x ... x bk x m x d`):
#                 Second set of data.
#             :attr:`diag` (bool):
#                 Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`.
#             :attr:`last_dim_is_batch` (tuple, optional):
#                 Is the last dimension of the data a batch dimension or not?
#             :attr:`square_dist` (bool):
#                 Should we square the distance matrix before returning?
#         Returns:
#             (:class:`Tensor`, :class:`Tensor) corresponding to the distance matrix between `x1` and `x2`.
#             The shape depends on the kernel's mode
#             * `diag=False`
#             * `diag=False` and `last_dim_is_batch=True`: (`b x d x n x n`)
#             * `diag=True`
#             * `diag=True` and `last_dim_is_batch=True`: (`b x d x n`)
#         """
#         if last_dim_is_batch:
#             x1 = x1.transpose(-1, -2).unsqueeze(-1)
#             x2 = x2.transpose(-1, -2).unsqueeze(-1)
#
#         x1_eq_x2 = torch.equal(x1, x2)
#
#         # torch scripts expect tensors
#         postprocess = torch.tensor(postprocess)
#
#         res = None
#
#         # Cache the Distance object or else JIT will recompile every time
#         if not self.distance_module or self.distance_module._postprocess != dist_postprocess_func:
#             self.distance_module = Distance(dist_postprocess_func)
#
#         if diag:
#             # Special case the diagonal because we can return all zeros most of the time.
#             if x1_eq_x2:
#                 res = torch.zeros(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
#                 if postprocess:
#                     res = dist_postprocess_func(res)
#                 return res
#             else:
#                 res = torch.norm(x1 - x2, p=2, dim=-1)
#                 if square_dist:
#                     res = res.pow(2)
#             if postprocess:
#                 res = dist_postprocess_func(res)
#             return res
#
#         elif square_dist:
#             res = self.distance_module._sq_dist(x1, x2, postprocess, x1_eq_x2)
#         else:
#             res = self.distance_module._dist(x1, x2, postprocess, x1_eq_x2)
#
#         return res
#
#     def named_sub_kernels(self):
#         for name, module in self.named_modules():
#             if module is not self and isinstance(module, Kernel):
#                 yield name, module
#
#     def num_outputs_per_input(self, x1, x2):
#         """
#         How many outputs are produced per input (default 1)
#         if x1 is size `n x d` and x2 is size `m x d`, then the size of the kernel
#         will be `(n * num_outputs_per_input) x (m * num_outputs_per_input)`
#         Default: 1
#         """
#         return 1
#
#     def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
#         return exact_prediction_strategies.DefaultPredictionStrategy(
#             train_inputs, train_prior_dist, train_labels, likelihood
#         )
#
#     def sub_kernels(self):
#         for _, kernel in self.named_sub_kernels():
#             yield kernel
#
#     def __call__(self, x1, x2=None, diag=False, last_dim_is_batch=False, **params):
#         x1_, x2_ = x1, x2
#
#         # Select the active dimensions
#         if self.active_dims is not None:
#             x1_ = x1_.index_select(-1, self.active_dims)
#             if x2_ is not None:
#                 x2_ = x2_.index_select(-1, self.active_dims)
#
#         # Give x1_ and x2_ a last dimension, if necessary
#         if x1_.ndimension() == 1:
#             x1_ = x1_.unsqueeze(1)
#         if x2_ is not None:
#             if x2_.ndimension() == 1:
#                 x2_ = x2_.unsqueeze(1)
#             if not x1_.size(-1) == x2_.size(-1):
#                 raise RuntimeError("x1_ and x2_ must have the same number of dimensions!")
#
#         if x2_ is None:
#             x2_ = x1_
#
#         # Check that ard_num_dims matches the supplied number of dimensions
#         if settings.debug.on():
#             if self.ard_num_dims is not None and self.ard_num_dims != x1_.size(-1):
#                 raise RuntimeError(
#                     "Expected the input to have {} dimensionality "
#                     "(based on the ard_num_dims argument). Got {}.".format(self.ard_num_dims, x1_.size(-1))
#                 )
#
#         if diag:
#             res = super(Kernel, self).__call__(x1_, x2_, diag=True, last_dim_is_batch=last_dim_is_batch, **params)
#             # Did this Kernel eat the diag option?
#             # If it does not return a LazyEvaluatedKernelTensor, we can call diag on the output
#             if not isinstance(res, LazyEvaluatedKernelTensor):
#                 if res.dim() == x1_.dim() and res.shape[-2:] == torch.Size((x1_.size(-2), x2_.size(-2))):
#                     res = res.diag()
#             return res
#
#         else:
#             if settings.lazily_evaluate_kernels.on():
#                 res = LazyEvaluatedKernelTensor(x1_, x2_, kernel=self, last_dim_is_batch=last_dim_is_batch, **params)
#             else:
#                 res = lazify(super(Kernel, self).__call__(x1_, x2_, last_dim_is_batch=last_dim_is_batch, **params))
#             return res
#
#     def __getstate__(self):
#         # JIT ScriptModules cannot be pickled
#         self.distance_module = None
#         return self.__dict__
#
#     def __add__(self, other):
#         kernels = []
#         kernels += self.kernels if isinstance(self, AdditiveKernel) else [self]
#         kernels += other.kernels if isinstance(other, AdditiveKernel) else [other]
#         return AdditiveKernel(*kernels)
#
#     def __mul__(self, other):
#         kernels = []
#         kernels += self.kernels if isinstance(self, ProductKernel) else [self]
#         kernels += other.kernels if isinstance(other, ProductKernel) else [other]
#         return ProductKernel(*kernels)
#
#     def __setstate__(self, d):
#         self.__dict__ = d
#
#     def __getitem__(self, index):
#         if len(self.batch_shape) == 0:
#             return self
#
#         new_kernel = deepcopy(self)
#         # Process the index
#         index = index if isinstance(index, tuple) else (index,)
#
#         for param_name, param in self._parameters.items():
#             new_kernel._parameters[param_name].data = param.__getitem__(index)
#             ndim_removed = len(param.shape) - len(new_kernel._parameters[param_name].shape)
#             new_batch_shape_len = len(self.batch_shape) - ndim_removed
#             new_kernel.batch_shape = new_kernel._parameters[param_name].shape[:new_batch_shape_len]
#
#         for sub_module_name, sub_module in self.named_sub_kernels():
#             new_kernel._modules[sub_module_name] = sub_module.__getitem__(index)
#
#         return new_kernel
#
#
#
# class AdditiveKernel(Kernel):
#     """
#     A Kernel that supports summing over multiple component kernels.
#     Example:
#         >>> covar_module = RBFKernel(active_dims=torch.tensor([1])) + RBFKernel(active_dims=torch.tensor([2]))
#         >>> x1 = torch.randn(50, 2)
#         >>> additive_kernel_matrix = covar_module(x1)
#     """
#
#     @property
#     def is_stationary(self) -> bool:
#         """
#         Kernel is stationary if all components are stationary.
#         """
#         return all(k.is_stationary for k in self.kernels)
#
#     def __init__(self, *kernels):
#         super(AdditiveKernel, self).__init__()
#         self.kernels = ModuleList(kernels)
#
#     def forward(self, x1, x2, diag=False, **params):
#         res = ZeroLazyTensor() if not diag else 0
#         for kern in self.kernels:
#             next_term = kern(x1, x2, diag=diag, **params)
#             if not diag:
#                 res = res + lazify(next_term)
#             else:
#                 res = res + next_term
#
#         return res
#
#     def num_outputs_per_input(self, x1, x2):
#         return self.kernels[0].num_outputs_per_input(x1, x2)
#
#     def __getitem__(self, index):
#         new_kernel = deepcopy(self)
#         for i, kernel in enumerate(self.kernels):
#             new_kernel.kernels[i] = self.kernels[i].__getitem__(index)
#
#         return new_kernel
#
#
# class ProductKernel(Kernel):
#     """
#     A Kernel that supports elementwise multiplying multiple component kernels together.
#     Example:
#         >>> covar_module = RBFKernel(active_dims=torch.tensor([1])) * RBFKernel(active_dims=torch.tensor([2]))
#         >>> x1 = torch.randn(50, 2)
#         >>> kernel_matrix = covar_module(x1) # The RBF Kernel already decomposes multiplicatively, so this is foolish!
#     """
#
#     @property
#     def is_stationary(self) -> bool:
#         """
#         Kernel is stationary if all components are stationary.
#         """
#         return all(k.is_stationary for k in self.kernels)
#
#     def __init__(self, *kernels):
#         super(ProductKernel, self).__init__()
#         self.kernels = ModuleList(kernels)
#
#     def forward(self, x1, x2, diag=False, **params):
#         x1_eq_x2 = torch.equal(x1, x2)
#
#         if not x1_eq_x2:
#             # If x1 != x2, then we can't make a MulLazyTensor because the kernel won't necessarily be square/symmetric
#             res = delazify(self.kernels[0](x1, x2, diag=diag, **params))
#         else:
#             res = self.kernels[0](x1, x2, diag=diag, **params)
#
#             if not diag:
#                 res = lazify(res)
#
#         for kern in self.kernels[1:]:
#             next_term = kern(x1, x2, diag=diag, **params)
#             if not x1_eq_x2:
#                 # Again delazify if x1 != x2
#                 res = res * delazify(next_term)
#             else:
#                 if not diag:
#                     res = res * lazify(next_term)
#                 else:
#                     res = res * next_term
#
#         return res
#
#     def num_outputs_per_input(self, x1, x2):
#         return self.kernels[0].num_outputs_per_input(x1, x2)
#
#     def __getitem__(self, index):
#         new_kernel = deepcopy(self)
#         for i, kernel in enumerate(self.kernels):
#             new_kernel.kernels[i] = self.kernels[i].__getitem__(index)
#
#         return new_kernel


class MinKernel(Kernel):

    def __init__(
        self,
        offset_prior: Optional[Prior] = None,
        offset_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        super(MinKernel, self).__init__(**kwargs)
        if offset_constraint is None:
            offset_constraint = Interval(offset_lower, offset_upper, initial_value=offset_init)
        self.register_parameter(name="raw_offset", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))

        if offset_prior is not None:
            if not isinstance(offset_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(offset_prior).__name__)
            self.register_prior("offset_prior", offset_prior, lambda m: m.offset, lambda m, v: m._set_offset(v))

        self.register_constraint("raw_offset", offset_constraint)

    @property
    def offset(self) -> torch.Tensor:
        return self.raw_offset_constraint.transform(self.raw_offset)

    @offset.setter
    def offset(self, value: torch.Tensor) -> None:
        self._set_offset(value)

    def _set_offset(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_offset)
        self.initialize(raw_offset=self.raw_offset_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1_ = x1
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
        a = torch.ones(x1_.shape)
        # a.to(x1_.device)  # for cuda vs cpu  (I don't think this works - may need fixing later)
        offset = self.offset.view(*self.batch_shape, 1, 1)

        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLazyTensor when x1 == x2 for efficiency when composing
            # with other kernels
            aa = MatmulLazyTensor(x1_, a.transpose(-2, -1))
            bb = aa.transpose(-2,-1)
            K = 0.5 * (aa + bb - torch.abs((aa - bb).evaluate())) + offset

        else:
            x2_ = x2
            if last_dim_is_batch:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)
            b = torch.ones(x2_.shape)
            # b.to(x2_.device)  # for cuda vs cpu  (I don't think this works - may need fixing later)
            aa = MatmulLazyTensor(x1_, b.transpose(-2, -1))
            bb = MatmulLazyTensor(a, x2_.transpose(-2, -1))
            K = 0.5 * (aa + bb - torch.abs((aa - bb).evaluate())) + offset
        if diag:
            return K.diag()
        else:
            return K


def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()


class CauchyKernel(Kernel):
    has_lengthscale = True
    def __init__(self, **kwargs):
        super(CauchyKernel, self).__init__(**kwargs)
        lengthscale_constraint = Interval(l_chy_lower, l_chy_upper, initial_value=l_chy_init)
        self.register_constraint("raw_lengthscale", lengthscale_constraint)

    def forward(self, x1, x2, diag=False, **params):

        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)

        d = self.covar_dist(
            x1_, x2_, square_dist=True, diag=diag, **params
        )

        K = 1 / (1.0 + d)
        return K


class MehlerKernel(Kernel):

    has_lengthscale = True

    def __init__(self, **kwargs):
        super(MehlerKernel, self).__init__(**kwargs)
        self.register_constraint("raw_lengthscale", Interval(l_rbf_lower, l_rbf_upper, initial_value=l_rbf_init))

    def forward(self, x1, x2, diag=False, **params):
        ell = self.lengthscale
        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLazyTensor when x1 == x2 for efficiency when composing
            # with other kernels
            prod = RootLazyTensor(x1).evaluate()


        else:
            prod = MatmulLazyTensor(x1, x2.transpose(-2, -1)).evaluate()
        dist = self.covar_dist(x1, x2, square_dist=True, diag=diag, **params)
        arg = dist + prod.mul(2*(1-torch.sqrt(1+ell**2)))
        arg = arg.div(-2*ell**2)  # using probabilists form
        K = arg.exp_()
        #K = K.div((1-rho**2).sqrt())  # removed scalar

        if diag:
            return K.diag()
        else:
            return K


class AR2Kernel(Kernel):
    # Berlinet et. al p317

    has_lengthscale = False

    def __init__(
        self,
        lengthscale_prior: Optional[Prior] = None,
        lengthscale_constraint: Optional[Interval] = None,
        period_prior: Optional[Prior] = None,
        period_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        super(AR2Kernel, self).__init__(**kwargs)
        if lengthscale_constraint is None:
            lengthscale_constraint = Interval(torch.tensor(0.05),
                                                      torch.tensor(500),
                                                      initial_value = torch.tensor(10))
        if period_constraint is None:
            period_constraint = Interval(1e-4,10, initial_value=0.75)

        self.register_parameter(
            name="raw_lengthscale", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_parameter(
            name="raw_period", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        if lengthscale_prior is not None:
            if not isinstance(lengthscale_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(lengthscale_prior).__name__)
            self.register_prior(
                "lengthscale_prior",
                lengthscale_prior,
                lambda m: m.lengthscale,
                lambda m, v: m._set_lengthscale(v),
            )
        if period_prior is not None:
            if not isinstance(period_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(period_prior).__name__)
            self.register_prior(
                "period_prior",
                period_prior,
                lambda m: m.period,
                lambda m, v: m._set_period(v),
            )

        self.register_constraint("raw_lengthscale", lengthscale_constraint)
        self.register_constraint("raw_period", period_constraint)

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @property
    def period(self):
        return self.raw_period_constraint.transform(self.raw_period)

    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)
    def _set_lengthscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))

    @period.setter
    def period(self, value):
        self._set_period(value)

    def _set_period(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period)
        self.initialize(raw_period=self.raw_period_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        diff = self.covar_dist(x1, x2, diag=diag, **params)
        pi = torch.tensor(torch.pi)
        #gamma_sq = self.period.pow(2) + self.lengthscale.pow(2)
        a = torch.exp(-diff.div(self.lengthscale))
        b = torch.cos(diff.mul(pi).div(self.period))
        c = torch.sin(diff.mul(pi).div(self.period)).mul(self.period).div(self.lengthscale).div(pi)
        # const = self.lengthscale.div(4).div(torch.add(
        #     torch.tensor(1).div(self.lengthscale.pow(2)),
        #     pi.pow(2).div(self.period.pow(2))))

        #res = a.mul(b).mul(const) + c
        res = (b+c).mul(a)
        if diag:
            res = res.squeeze(0)
        return res





# class AR2Kernel(Kernel):
#     # Berlinet et. al p317
#
#     has_lengthscale = False
#
#     def __init__(
#         self,
#         alpha_prior: Optional[Prior] = None,
#         alpha_constraint: Optional[Interval] = None,
#         omega_prior: Optional[Prior] = None,
#         omega_constraint: Optional[Interval] = None,
#         **kwargs,
#     ):
#         super(AR2Kernel, self).__init__(**kwargs)
#         if alpha_constraint is None:
#             alpha_constraint = Interval(1e-4,0.5, initial_value=0.1)
#         if omega_constraint is None:
#             omega_constraint = Interval(1e-4,10, initial_value=0.5)
#
#         self.register_parameter(
#             name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
#         )
#         self.register_parameter(
#             name="raw_omega", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
#         )
#
#         if alpha_prior is not None:
#             if not isinstance(alpha_prior, Prior):
#                 raise TypeError("Expected gpytorch.priors.Prior but got " + type(alpha_prior).__name__)
#             self.register_prior(
#                 "alpha_prior",
#                 alpha_prior,
#                 lambda m: m.alpha,
#                 lambda m, v: m._set_alpha(v),
#             )
#         if omega_prior is not None:
#             if not isinstance(omega_prior, Prior):
#                 raise TypeError("Expected gpytorch.priors.Prior but got " + type(omega_prior).__name__)
#             self.register_prior(
#                 "omega_prior",
#                 omega_prior,
#                 lambda m: m.omega,
#                 lambda m, v: m._set_omega(v),
#             )
#
#         self.register_constraint("raw_alpha", alpha_constraint)
#         self.register_constraint("raw_omega", omega_constraint)
#
#     @property
#     def alpha(self):
#         return self.raw_alpha_constraint.transform(self.raw_alpha)
#
#     @property
#     def omega(self):
#         return self.raw_omega_constraint.transform(self.raw_omega)
#
#     @alpha.setter
#     def alpha(self, value):
#         self._set_alpha(value)
#     def _set_alpha(self, value):
#         if not torch.is_tensor(value):
#             value = torch.as_tensor(value).to(self.raw_alpha)
#         self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))
#
#     @omega.setter
#     def omega(self, value):
#         self._set_omega(value)
#
#     def _set_omega(self, value):
#         if not torch.is_tensor(value):
#             value = torch.as_tensor(value).to(self.raw_omega)
#         self.initialize(raw_omega=self.raw_omega_constraint.inverse_transform(value))
#
#     def forward(self, x1, x2, diag=False, **params):
#         diff = self.covar_dist(x1, x2, diag=diag, **params)
#         gamma_sq = self.omega.pow(2) + self.alpha.pow(2)
#         a = torch.exp(-diff.mul(self.alpha)).div(4).div(self.alpha).div(gamma_sq)
#         b = torch.cos(diff.mul(self.omega))
#         c = torch.sin(diff.mul(self.omega)).mul(self.alpha).div(self.omega)
#
#         res = a.mul(b) + c
#         if diag:
#             res = res.squeeze(0)
#         return res










# Old Kernels with constraints





class CosineKernel(Kernel):
    is_stationary = True

    def __init__(
        self,
        period_length_prior: Optional[Prior] = None,
        period_length_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        super(CosineKernel, self).__init__(**kwargs)

        self.register_parameter(
            name="raw_period_length", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        if period_length_constraint is None:
            period_length_constraint = Interval(per_lower, per_upper, initial_value = per_init)

        if period_length_prior is not None:
            if not isinstance(period_length_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(period_length_prior).__name__)
            self.register_prior(
                "period_length_prior",
                period_length_prior,
                lambda m: m.period_length,
                lambda m, v: m._set_period_length(v),
            )

        self.register_constraint("raw_period_length", period_length_constraint)

    @property
    def period_length(self):
        return self.raw_period_length_constraint.transform(self.raw_period_length)

    @period_length.setter
    def period_length(self, value):
        return self._set_period_length(value)

    def _set_period_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period_length)

        self.initialize(raw_period_length=self.raw_period_length_constraint.inverse_transform(value))

    def forward(self, x1, x2, **params):
        x1_ = x1.div(self.period_length)
        x2_ = x2.div(self.period_length)
        diff = self.covar_dist(x1_, x2_, **params)
        res = torch.cos(diff.mul(math.pi))
        return res


class MaternKernel(Kernel):
    has_lengthscale = True

    def __init__(self, nu: Optional[float] = 2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MaternKernel, self).__init__(**kwargs)
        self.nu = nu
        if nu == 0.5:
            lengthscale_constraint = Interval(l_m12_lower, l_m12_upper, initial_value=l_m12_init)
        elif nu == 1.5:
            lengthscale_constraint = Interval(l_m32_lower, l_m32_upper, initial_value=l_m32_init)
        else:
            lengthscale_constraint = Interval(l_m52_lower, l_m52_upper, initial_value=l_m52_init)

        self.register_constraint("raw_lengthscale", lengthscale_constraint)

    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

            x1_ = (x1 - mean).div(self.lengthscale)
            x2_ = (x2 - mean).div(self.lengthscale)
            distance = self.covar_dist(x1_, x2_, diag=diag, **params)
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            else:
                constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
            return constant_component * exp_component
        return MaternCovariance.apply(
            x1, x2, self.lengthscale, self.nu, lambda x1, x2: self.covar_dist(x1, x2, **params)
        )


class LinearKernel(Kernel):
    def __init__(
        self,
        num_dimensions: Optional[int] = None,
        offset_prior: Optional[Prior] = None,
        variance_prior: Optional[Prior] = None,
        variance_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        super(LinearKernel, self).__init__(**kwargs)
        if variance_constraint is None:
            variance_constraint = Positive()

        if num_dimensions is not None:
            # Remove after 1.0
            warnings.warn("The `num_dimensions` argument is deprecated and no longer used.", DeprecationWarning)
            self.register_parameter(name="offset", parameter=torch.nn.Parameter(torch.zeros(1, 1, num_dimensions)))
        if offset_prior is not None:
            # Remove after 1.0
            warnings.warn("The `offset_prior` argument is deprecated and no longer used.", DeprecationWarning)
        self.register_parameter(name="raw_variance", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
        if variance_prior is not None:
            if not isinstance(variance_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(variance_prior).__name__)
            self.register_prior("variance_prior", variance_prior, lambda m: m.variance, lambda m, v: m._set_variance(v))

        self.register_constraint("raw_variance", variance_constraint)

    @property
    def variance(self):
        return self.raw_variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value):
        self._set_variance(value)

    def _set_variance(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_variance)
        self.initialize(raw_variance=self.raw_variance_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1_ = x1 * self.variance.sqrt()
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)

        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLazyTensor when x1 == x2 for efficiency when composing
            # with other kernels
            prod = RootLazyTensor(x1_)

        else:
            x2_ = x2 * self.variance.sqrt()
            if last_dim_is_batch:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)

            prod = MatmulLazyTensor(x1_, x2_.transpose(-2, -1))

        if diag:
            return prod.diag()
        else:
            return prod

class PeriodicKernel(Kernel):
    has_lengthscale = True

    def __init__(
        self,
        period_length_prior: Optional[Prior] = None,
        period_length_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        super(PeriodicKernel, self).__init__(**kwargs)
        if period_length_constraint is None:
            period_length_constraint = Interval(per_lower, per_upper, initial_value=per_init)
        lengthscale_constraint = Interval(l_rbf_lower, l_rbf_upper, initial_value=l_rbf_init)
        self.register_constraint("raw_lengthscale", lengthscale_constraint)

        ard_num_dims = kwargs.get("ard_num_dims", 1)
        self.register_parameter(
            name="raw_period_length", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, ard_num_dims))
        )

        if period_length_prior is not None:
            if not isinstance(period_length_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(period_length_prior).__name__)
            self.register_prior(
                "period_length_prior",
                period_length_prior,
                lambda m: m.period_length,
                lambda m, v: m._set_period_length(v),
            )

        self.register_constraint("raw_period_length", period_length_constraint)

    @property
    def period_length(self):
        return self.raw_period_length_constraint.transform(self.raw_period_length)

    @period_length.setter
    def period_length(self, value):
        self._set_period_length(value)

    def _set_period_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period_length)
        self.initialize(raw_period_length=self.raw_period_length_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        # Pop this argument so that we can manually sum over dimensions
        last_dim_is_batch = params.pop("last_dim_is_batch", False)
        # Get lengthscale
        lengthscale = self.lengthscale

        x1_ = x1.div(self.period_length / math.pi)
        x2_ = x2.div(self.period_length / math.pi)
        # We are automatically overriding last_dim_is_batch here so that we can manually sum over dimensions.
        diff = self.covar_dist(x1_, x2_, diag=diag, last_dim_is_batch=True, **params)

        if diag:
            lengthscale = lengthscale[..., 0, :, None]
        else:
            lengthscale = lengthscale[..., 0, :, None, None]
        exp_term = diff.sin().pow(2.0).div(lengthscale).mul(-2.0)

        if not last_dim_is_batch:
            exp_term = exp_term.sum(dim=(-2 if diag else -3))

        return exp_term.exp()

class PolynomialKernel(Kernel):
    def __init__(
        self,
        power: int,
        offset_prior: Optional[Prior] = None,
        offset_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        super(PolynomialKernel, self).__init__(**kwargs)
        if offset_constraint is None:
            offset_constraint = Interval(offset_lower, offset_upper, initial_value=offset_init)

        self.register_parameter(name="raw_offset", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))

        # We want the power to be a float so we dont have to worry about its device / dtype.
        if torch.is_tensor(power):
            if power.numel() > 1:
                raise RuntimeError("Cant create a Polynomial kernel with more than one power")
            else:
                power = power.item()

        self.power = power

        if offset_prior is not None:
            if not isinstance(offset_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(offset_prior).__name__)
            self.register_prior("offset_prior", offset_prior, lambda m: m.offset, lambda m, v: m._set_offset(v))

        self.register_constraint("raw_offset", offset_constraint)

    @property
    def offset(self) -> torch.Tensor:
        return self.raw_offset_constraint.transform(self.raw_offset)

    @offset.setter
    def offset(self, value: torch.Tensor) -> None:
        self._set_offset(value)

    def _set_offset(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_offset)
        self.initialize(raw_offset=self.raw_offset_constraint.inverse_transform(value))

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: Optional[bool] = False,
        last_dim_is_batch: Optional[bool] = False,
        **params,
    ) -> torch.Tensor:
        offset = self.offset.view(*self.batch_shape, 1, 1)

        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        if diag:
            return ((x1 * x2).sum(dim=-1) + self.offset).pow(self.power)

        if (x1.dim() == 2 and x2.dim() == 2) and offset.dim() == 2:
            return torch.addmm(offset, x1, x2.transpose(-2, -1)).pow(self.power)
        else:
            return (torch.matmul(x1, x2.transpose(-2, -1)) + offset).pow(self.power)


def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()


class RBFKernel(Kernel):
    has_lengthscale = True
    def __init__(
        self,
        **kwargs,
    ):
        super(RBFKernel, self).__init__(**kwargs)
        lengthscale_constraint = Interval(l_rbf_lower, l_rbf_upper, initial_value=l_rbf_init)
        self.register_constraint("raw_lengthscale", lengthscale_constraint)

    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
            return self.covar_dist(
                x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rbf, postprocess=True, **params
            )
        return RBFCovariance.apply(
            x1,
            x2,
            self.lengthscale,
            lambda x1, x2: self.covar_dist(
                x1, x2, square_dist=True, diag=False, dist_postprocess_func=postprocess_rbf, postprocess=False, **params
            ),
        )


class ScaleKernel(Kernel):
    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if base kernel is stationary.
        """
        return self.base_kernel.is_stationary

    def __init__(
        self,
        base_kernel: Kernel,
        outputscale_prior: Optional[Prior] = None,
        outputscale_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims
        super(ScaleKernel, self).__init__(**kwargs)
        if outputscale_constraint is None:
            outputscale_constraint = Interval(c_lower, c_upper, initial_value=c_init)

        self.base_kernel = base_kernel
        outputscale = torch.zeros(*self.batch_shape) if len(self.batch_shape) else torch.tensor(0.0)
        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(outputscale))
        if outputscale_prior is not None:
            if not isinstance(outputscale_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(outputscale_prior).__name__)
            self.register_prior(
                "outputscale_prior", outputscale_prior, self._outputscale_param, self._outputscale_closure
            )

        self.register_constraint("raw_outputscale", outputscale_constraint)

    def _outputscale_param(self, m):
        return m.outputscale

    def _outputscale_closure(self, m, v):
        return m._set_outputscale(v)

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value):
        self._set_outputscale(value)

    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    def forward(self, x1, x2, last_dim_is_batch=False, diag=False, **params):
        orig_output = self.base_kernel.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        outputscales = self.outputscale
        if last_dim_is_batch:
            outputscales = outputscales.unsqueeze(-1)
        if diag:
            outputscales = outputscales.unsqueeze(-1)
            return delazify(orig_output) * outputscales
        else:
            outputscales = outputscales.view(*outputscales.shape, 1, 1)
            return orig_output.mul(outputscales)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return self.base_kernel.prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood)