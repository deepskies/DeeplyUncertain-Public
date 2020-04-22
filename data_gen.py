import numpy as np
from scipy.special import gamma


def pendulum(n=10000, n_t=10, g_range=None, theta_range=None, ell_range=None, m_range=None, t_spread=None,
             ell_spread=None, seed=42):
    if g_range is None:
        g_range = [5, 15]
    if theta_range is None:
        theta_range = [5, 15]
    if ell_range is None:
        ell_range = [0.2, 0.8]
    if m_range is None:
        m_range = [0.02, 0.1]
    if t_spread is None:
        t_spread = [0.03, 0.03]
    if ell_spread is None:
        ell_spread = [0., 0.]

    np.random.seed(seed)

    g = (g_range[1] - g_range[0]) * np.random.rand(n) + g_range[0]
    theta = ((theta_range[1] - theta_range[0]) * np.random.rand(n) + theta_range[0]).reshape((n, 1))
    ell = ((ell_range[1] - ell_range[0]) * np.random.rand(n) + ell_range[0]).reshape((n, 1))
    m = ((m_range[1] - m_range[0]) * np.random.rand(n) + m_range[0]).reshape((n, 1))
    t = (2 * np.pi * np.sqrt(ell / g.reshape((n, 1))))

    t_scales = np.random.uniform(t_spread[0], t_spread[1], n)
    t_scales = np.repeat(t_scales, n_t).reshape((n, n_t))
    t_spreads = np.random.normal(scale=t_scales, size=(n, n_t))
    t_sigma = t_spreads * t
    t = t + t_sigma

    ell_scales = np.random.uniform(ell_spread[0], ell_spread[1], n).reshape((n, 1))
    ell_spreads = np.random.normal(scale=ell_scales, size=(n, 1))
    ell_sigma = ell_spreads * ell
    ell = ell + ell_sigma

    feat = np.concatenate([theta, ell, m, t], axis=1)
    y = g

    # statistical uncertainty
    delta_t = np.std(t_sigma, axis=1) / np.sqrt(2) * gamma((n_t-1)/2) / gamma(n_t/2)
    mean_t = np.mean(t, axis=1)
    ell = ell.reshape(n, )
    delta_ell = ell * ell_scales.reshape(n, )
    calc_y = 4 * np.pi ** 2 * ell / mean_t ** 2
    delta_y = 4 * np.pi ** 2 / mean_t ** 2 * np.sqrt((2 * ell * delta_t / mean_t) ** 2 + delta_ell ** 2)

    return feat, y, calc_y, delta_y
