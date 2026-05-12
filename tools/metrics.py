import warnings

import numpy as np
from scipy import linalg


def _safe_sqrtm(matrix):
    """Wrap scipy.linalg.sqrtm so that non-finite inputs or convergence
    failures are reported as a non-finite covmean instead of raising.

    Returns a numpy array of the same shape as ``matrix``. When the
    underlying sqrtm fails for any reason, the returned array is filled
    with ``np.inf`` so the caller can detect the failure via
    ``np.isfinite(covmean).all()``.
    """
    if not np.isfinite(matrix).all():
        return np.full_like(matrix, np.inf, dtype=np.float64)
    try:
        covmean, _ = linalg.sqrtm(matrix, disp=False)
    except (ValueError, linalg.LinAlgError) as err:
        warnings.warn(
            "sqrtm failed ({}); treating covmean as non-finite.".format(err)
        )
        return np.full_like(matrix, np.inf, dtype=np.float64)
    return covmean


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance. Returns ``np.inf`` when the computation is
           not numerically feasible (e.g. NaN/Inf inputs or sqrtm failure).
    """

    mu1 = np.atleast_1d(mu1).astype(np.float64, copy=False)
    mu2 = np.atleast_1d(mu2).astype(np.float64, copy=False)

    sigma1 = np.atleast_2d(sigma1).astype(np.float64, copy=False)
    sigma2 = np.atleast_2d(sigma2).astype(np.float64, copy=False)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    # Guard 1: if inputs themselves contain non-finite values, sanitise them
    # so we can still produce a numerical answer (or fall back to inf below).
    inputs_finite = (
        np.isfinite(mu1).all()
        and np.isfinite(mu2).all()
        and np.isfinite(sigma1).all()
        and np.isfinite(sigma2).all()
    )
    if not inputs_finite:
        warnings.warn(
            "calculate_frechet_distance received non-finite mu/sigma; "
            "sanitising with nan_to_num and adding eps*I to covariances."
        )
        mu1 = np.nan_to_num(mu1, nan=0.0, posinf=0.0, neginf=0.0)
        mu2 = np.nan_to_num(mu2, nan=0.0, posinf=0.0, neginf=0.0)
        sigma1 = np.nan_to_num(sigma1, nan=0.0, posinf=0.0, neginf=0.0)
        sigma2 = np.nan_to_num(sigma2, nan=0.0, posinf=0.0, neginf=0.0)
        sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
        sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2

    covmean = _safe_sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = _safe_sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if not np.isfinite(covmean).all():
            warnings.warn(
                "calculate_frechet_distance: sqrtm still non-finite after "
                "offset; returning np.inf."
            )
            return float("inf")

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            warnings.warn(
                "calculate_frechet_distance: large imaginary component {}; "
                "returning np.inf.".format(m)
            )
            return float("inf")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid_value = (diff.dot(diff) + np.trace(sigma1) +
                 np.trace(sigma2) - 2 * tr_covmean)
    if not np.isfinite(fid_value):
        warnings.warn(
            "calculate_frechet_distance produced a non-finite result; "
            "returning np.inf."
        )
        return float("inf")
    return float(fid_value)
