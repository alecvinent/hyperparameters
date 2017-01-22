import operator as op
import itertools as it
import numpy as np
from functools import partial
from collections import deque
from core import grad, Differentiable
#from exact_rep import ExactRep
from scipy.optimize import minimize
from nn_utils import fill_parser

RADIX_SCALE = 2**52

def sgd_numpy_safe(L_grad, meta, x0, alpha, gamma, N_iters, x_init,
                callback=None, forward_pass_only=True):
    N_safe_sampling = len(alpha) #1000/10
    x_final = x0
    if forward_pass_only:
        return x_final

    def hypergrad(outgrad):
        d_x = outgrad
        d_v, d_meta = np.zeros(d_x.shape), np.zeros(meta.shape)
        grad_proj = lambda x, meta, d, i: np.dot(L_grad(x, meta, i), d)
        L_hvp_x = grad(grad_proj, 0)  # Returns a size(x) output.
        L_hvp_meta = grad(grad_proj, 1)  # Returns a size(meta) output.
        beta = np.linspace(0.001,0.999,N_safe_sampling)
        for i in range(N_safe_sampling)[::-1]:
            x_current = (1-beta[i])*x_init + beta[i]*x_final
            d_alphas = np.dot(d_x, d_v)
            d_v += d_x * alpha
            d_x -= (1.0 - gamma) * L_hvp_x(x_current, meta, d_v, i)
            d_meta -= (1.0 - gamma) * L_hvp_meta(x_current, meta, d_v, i)
            d_v *= gamma
        # assert np.all(ExactRep(x0).val == X.val)
        return d_meta, d_alphas

    return x_final, [None, hypergrad]


sgd_numpy_safe = Differentiable(sgd_numpy_safe,
                               partial(sgd_numpy_safe, forward_pass_only=False))
