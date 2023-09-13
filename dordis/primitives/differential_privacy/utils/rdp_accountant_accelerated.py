from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing import cpu_count, Pool
import math
import numpy as np

processes = cpu_count()


def _compute_delta_tool(a, r, eps):
  if a < 1:
    raise ValueError("Renyi divergence order must be >=1.")
  if r < 0:
    raise ValueError("Renyi divergence must be >=0.")

  logdelta = 0.5 * math.log1p(-math.exp(-r))
  if a > 1.01:
    rdp_bound = (a - 1) * (r - eps + math.log1p(-1 / a)) - math.log(a)
    logdelta = min(logdelta, rdp_bound)

  return logdelta


def _compute_delta(orders, rdp, eps):
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if eps < 0:
    raise ValueError("Value of privacy loss bound epsilon must be >=0.")
  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  # logdeltas = []
  # for (a, r) in zip(orders_vec, rdp_vec):
  #   if a < 1:
  #     raise ValueError("Renyi divergence order must be >=1.")
  #   if r < 0:
  #     raise ValueError("Renyi divergence must be >=0.")
  #
  #   logdelta = 0.5 * math.log1p(-math.exp(-r))
  #   if a > 1.01:
  #     rdp_bound = (a - 1) * (r - eps + math.log1p(-1 / a)) - math.log(a)
  #     logdelta = min(logdelta, rdp_bound)
  #
  #   logdeltas.append(logdelta)

  input_list = [(a, r, eps) for a, r in zip(orders_vec, rdp_vec)]
  with Pool(processes=processes) as pool:
    logdeltas = pool.starmap(_compute_delta_tool, input_list)

  idx_opt = np.argmin(logdeltas)
  return min(math.exp(logdeltas[idx_opt]), 1.), orders_vec[idx_opt]


def _compute_eps_tool(a, r, delta):
  if a < 1:
    raise ValueError("Renyi divergence order must be >=1.")
  if r < 0:
    raise ValueError("Renyi divergence must be >=0.")

  if delta ** 2 + math.expm1(-r) >= 0:
    eps = 0
  elif a > 1.01:
    eps = r + math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
  else:
    eps = np.inf

  return eps


def _compute_eps(orders, rdp, delta):
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if delta <= 0:
    raise ValueError("Privacy failure probability bound delta must be >0.")
  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  # eps_vec = []
  # for (a, r) in zip(orders_vec, rdp_vec):
  #   if a < 1:
  #     raise ValueError("Renyi divergence order must be >=1.")
  #   if r < 0:
  #     raise ValueError("Renyi divergence must be >=0.")
  #
  #   if delta**2 + math.expm1(-r) >= 0:
  #     eps = 0
  #   elif a > 1.01:
  #     eps = r + math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
  #   else:
  #     eps = np.inf
  #   eps_vec.append(eps)

  input_list = [(a, r, delta) for a, r in zip(orders_vec, rdp_vec)]
  with Pool(processes=processes) as pool:
    eps_vec = pool.starmap(_compute_eps_tool, input_list)

  idx_opt = np.argmin(eps_vec)
  return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]


def get_privacy_spent(orders, rdp, target_eps=None, target_delta=None):
  if target_eps is None and target_delta is None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (Both are).")

  if target_eps is not None and target_delta is not None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (None is).")

  if target_eps is not None:
    delta, opt_order = _compute_delta(orders, rdp, target_eps)
    return target_eps, delta, opt_order
  else:
    eps, opt_order = _compute_eps(orders, rdp, target_delta)
    return eps, target_delta, opt_order
