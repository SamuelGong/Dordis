import math
import numpy as np
from scipy import optimize, special
from dordis.primitives.differential_privacy.utils\
    .rdp_accountant import get_privacy_spent

RDP_ORDERS = tuple(range(2, 129)) + (256,)
DIV_EPSILON = 1e-22


def log_comb(n, k):
    gammaln = special.gammaln
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def _compute_rdp_subsampled(alpha, gamma, eps, upper_bound=True):
    if isinstance(alpha, float):
        assert alpha.is_integer()
        alpha = int(alpha)
    assert alpha > 1
    assert 0 < gamma <= 1

    if upper_bound:
        a = [0, eps(2)]
        b = [((1 - gamma)**(alpha - 1)) * (alpha * gamma - gamma + 1),
             special.comb(alpha, 2) * (gamma**2) * (1 - gamma)**(alpha - 2)]

        for l in range(3, alpha + 1):
            a.append((l - 1) * eps(l) + log_comb(alpha, l)
                     + (alpha - l) * np.log(1 - gamma) + l * np.log(gamma))
            b.append(3)

    else:
        a = [0]
        b = [((1 - gamma)**(alpha - 1)) * (alpha * gamma - gamma + 1)]

        for l in range(2, alpha + 1):
            a.append((l - 1) * eps(l) + log_comb(alpha, l)
                     + (alpha - l) * np.log(1 - gamma) + l * np.log(gamma))
            b.append(1)

    return special.logsumexp(a=a, b=b) / (alpha - 1)


def rounded_l1_norm_bound(l2_norm_bound, dim):
    return l2_norm_bound * min(np.sqrt(dim), l2_norm_bound)


def rounded_l2_norm_bound(l2_norm_bound, beta, dim):
    assert int(dim) == dim and dim > 0, f'Invalid dimension: {dim}'
    assert 0 <= beta < 1, 'beta must be in the range [0, 1)'
    assert l2_norm_bound > 0, 'Input l2_norm_bound should be positive.'

    bound_1 = l2_norm_bound + np          .sqrt(dim)
    if beta == 0:
        return bound_1

    squared_bound_2 = np.square(l2_norm_bound) + 0.25 * dim
    squared_bound_2 += (
        np.sqrt(2.0 * np.log(1.0 / beta)) * (l2_norm_bound + 0.5 * np.sqrt(dim))
    )

    bound_2 = np.sqrt(squared_bound_2)
    return min(bound_1, bound_2)


def compute_rdp_dgaussian(q, l1_scale, l2_scale, tau, dim, steps, orders):
    orders = [int(order) for order in orders]

    def eps(order):
        assert order > 1, 'alpha must be greater than 1.'
        term_1 = (order / 2.0) * l2_scale**2 + tau * dim
        term_2 = (order / 2.0) * (l2_scale**2 + 2 * l1_scale * tau + tau**2 * dim)
        term_3 = (order / 2.0) * (l2_scale + np.sqrt(dim) * tau)**2
        return min(term_1, term_2, term_3)

    if q == 1:
        rdp = np.array([eps(order) for order in orders])
    else:
        rdp = np.array([
            min(_compute_rdp_subsampled(order, q, eps), eps(order))
            for order in orders
        ])
    return rdp * steps


def ddgauss_epsilon(gamma,
                    local_stddev,
                    num_clients,
                    l2_sens,
                    beta,
                    dim,
                    q,
                    steps,
                    delta,
                    l1_sens=None,
                    rounding=True,
                    orders=RDP_ORDERS):
    scale = 1.0 / (gamma + DIV_EPSILON)
    l1_sens = l1_sens or (l2_sens * np.sqrt(dim))
    if rounding:
        l2_sens = rounded_l2_norm_bound(l2_sens * scale, beta, dim) / scale
        l1_sens = rounded_l1_norm_bound(l2_sens * scale, dim) / scale

    tau = 0
    for k in range(1, num_clients):
        tau += math.exp(-2 * (math.pi * local_stddev * scale)**2 * (k / (k + 1)))
    tau *= 10

    # central stddev: np.sqrt(num_clients) * local_stddev
    l1_scale = l1_sens / (np.sqrt(num_clients) * local_stddev)
    l2_scale = l2_sens / (np.sqrt(num_clients) * local_stddev)
    rdp = compute_rdp_dgaussian(q, l1_scale, l2_scale, tau, dim, steps, orders)
    eps, _, order = get_privacy_spent(orders, rdp, target_delta=delta)
    return eps, order


def ddgauss_local_stddev(q,
                         epsilon,
                         l2_clip_norm,
                         gamma,
                         beta,
                         steps,
                         num_clients,
                         dim,
                         delta,
                         orders=RDP_ORDERS):
    def stddev_opt_fn(stddev):
        stddev += DIV_EPSILON
        cur_epsilon, _ = ddgauss_epsilon(
            gamma,
            stddev,
            num_clients,
            l2_clip_norm,
            beta,
            dim,
            q,
            steps,
            delta,
            orders=orders)
        return (epsilon - cur_epsilon) ** 2

    stddev_result = optimize.minimize_scalar(stddev_opt_fn)
    if stddev_result.success:
        return stddev_result.x
    else:
        return -1


def ddgauss_params(q,
                   epsilon,
                   l2_clip_norm,
                   bits,
                   num_clients,
                   dim,
                   delta,
                   beta,
                   steps,
                   k=4,
                   rho=1,
                   sqrtn_norm_growth=False,
                   orders=RDP_ORDERS):
    n_factor = num_clients ** (1 if sqrtn_norm_growth else 2)

    def stddev(gamma):
        return ddgauss_local_stddev(
            q=q, epsilon=epsilon, l2_clip_norm=l2_clip_norm,
            gamma=gamma, beta=beta, steps=steps, num_clients=num_clients,
            dim=dim, delta=delta, orders=orders
        )

    def mod_min(gamma):
        var = rho / dim * l2_clip_norm ** 2 * n_factor
        var += (gamma ** 2 / 4.0 + stddev(gamma) ** 2) * num_clients
        return k * math.sqrt(var)

    def gamma_opt_fn(gamma):
        return (math.pow(2, bits) - 2 * mod_min(gamma) / (gamma + DIV_EPSILON)) ** 2

    gamma_result = optimize.minimize_scalar(gamma_opt_fn)
    if not gamma_result.success:
        raise ValueError('Cannot compute gamma.')

    gamma = gamma_result.x
    local_stddev = ddgauss_local_stddev(
            q=q, epsilon=epsilon, l2_clip_norm=l2_clip_norm,
            gamma=gamma, beta=beta, steps=steps, num_clients=num_clients,
            dim=dim, delta=delta, orders=orders
        )
    return gamma, local_stddev


########################


def _skellam_rdp(l1_sens, l2_sens, central_var, scale, order):
    assert order > 1, f'alpha must be greater than 1. Found {order}.'
    a, s, mu = order, scale, central_var
    rdp = a / (2 * mu) * l2_sens ** 2
    rdp += min(((2 * a - 1) * s * l2_sens ** 2 + 6 * l1_sens) / (4 * s ** 3 * mu ** 2),
               3 * l1_sens / (2 * s * mu))
    return rdp


def skellam_rdp(
        scale,
        central_stddev,
        l2_sens,
        beta,
        dim,
        q,
        steps,
        l1_sens=None,
        rounding=True,
        orders=RDP_ORDERS):

    l1_sens = l1_sens or (l2_sens * np.sqrt(dim))
    if rounding:
        l2_sens = rounded_l2_norm_bound(l2_sens * scale, beta, dim) / scale
        l1_sens = rounded_l1_norm_bound(l2_sens * scale, dim) / scale

    orders = [int(order) for order in orders]
    central_var = central_stddev ** 2

    def eps_fn(order):
        return _skellam_rdp(l1_sens, l2_sens, central_var, scale, order)

    if q == 1:
        rdp = np.array([eps_fn(order) for order in orders])
    else:
        rdp = np.array([
            min(_compute_rdp_subsampled(order, q, eps_fn), eps_fn(order))
            for order in orders
        ])
    return rdp * steps


def skellam_epsilon(
        scale,
        central_stddev,
        l2_sens,
        beta,
        dim,
        q,
        steps,
        delta,
        orders=RDP_ORDERS):

    rdp = skellam_rdp(
        scale=scale,
        central_stddev=central_stddev,
        l2_sens=l2_sens,
        beta=beta,
        dim=dim,
        q=q,
        steps=steps,
        orders=orders
    )
    eps, _, order = get_privacy_spent(orders, rdp, target_delta=delta)
    return eps, order


def dskellam_local_stddev(epsilon,
                         scale,
                         l2_clip_norm,
                         num_clients,
                         beta,
                         dim,
                         q,
                         steps,
                         delta,
                         orders=RDP_ORDERS):

    def stddev_opt_fn(local_stddev):
        local_stddev += DIV_EPSILON
        central_stddev = local_stddev * np.sqrt(num_clients)
        cur_epsilon, _ = skellam_epsilon(
            scale=scale,
            central_stddev=central_stddev,
            l2_sens=l2_clip_norm,
            beta=beta,
            dim=dim,
            q=q,
            steps=steps,
            delta=delta,
            orders=orders
        )
        return (epsilon - cur_epsilon) ** 2

    local_stddev_result = optimize.minimize_scalar(stddev_opt_fn)
    if local_stddev_result.success:
        return local_stddev_result.x
    else:
        raise ValueError('Cannot compute local_stddev for Skellam.')


def dskellam_params(q,
                    epsilon,
                    l2_clip_norm,
                    bits,
                    num_clients,
                    dim,
                    delta,
                    beta,
                    steps,
                    k=3,
                    rho=1,
                    sqrtn_norm_growth=False,
                    orders=RDP_ORDERS):

    n_factor = num_clients ** (1 if sqrtn_norm_growth else 2)

    def local_stddev(gamma):
        scale = 1.0 / (gamma + DIV_EPSILON)
        return dskellam_local_stddev(
            q=q, epsilon=epsilon, l2_clip_norm=l2_clip_norm,
            scale=scale, beta=beta, steps=steps, num_clients=num_clients,
            dim=dim, delta=delta, orders=orders
        )

    def mod_min(gamma):
        var = rho / dim * l2_clip_norm ** 2 * n_factor
        var += (gamma ** 2 / 4.0 + local_stddev(gamma) ** 2) * num_clients
        return k * math.sqrt(var)

    def gamma_opt_fn(gamma):
        return (math.pow(2, bits) - 2 * mod_min(gamma) / (gamma + DIV_EPSILON)) ** 2

    gamma_result = optimize.minimize_scalar(gamma_opt_fn)
    if not gamma_result.success:
        raise ValueError('Cannot compute gamma.')

    scale = 1. / gamma_result.x
    local_stddev = dskellam_local_stddev(
            q=q, epsilon=epsilon, l2_clip_norm=l2_clip_norm,
            scale=scale, beta=beta, steps=steps, num_clients=num_clients,
            dim=dim, delta=delta, orders=orders
        )

    return scale, local_stddev