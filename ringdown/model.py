__all__ = ["make_mchi_model", "make_mchi_aligned_model", "make_ftau_model"]

import pytensor.tensor as at
import pytensor.tensor.slinalg as atl
import numpy as np
import pymc as pm

import jax.numpy as jnp
import jax.scipy as jsp
import flowMC

# Typing guidelines : https://jax.readthedocs.io/en/latest/jax.typing.html
from jax import Array, jit
from jaxtyping import ArrayLike
from typing import Tuple

# reference frequency and mass values to translate linearly between the two
FREF = 2985.668287014743
MREF = 68.0

################################################################################
################################################################################
##########################   Jax Implementation    ##########################
################################################################################
################################################################################


def jax_mvnorm_cholesky_logpdf(y: ArrayLike, mu: ArrayLike, L: ArrayLike) -> Array:
    """Compute the log probability of y for a multivariate normal characterized by mean mu and cholesky decomposed covariance L

    Parameters
    ==========
    y : ArrayLike
        The input data for which the probability should be assessed
    mu : ArrayLike
        The mean of the multivariate norm
    L : ArrayLike
        The cholesky decomposition (lower triangular) of the distribution's covariance matrix

    Returns
    =======
    Array
        The log probability of y for this distribution
    """
    # Slightly modified for speed, jax compatibility and low number accuracy
    # from https://jeffpollock9.github.io/multivariate-normal-cholesky/
    d = jnp.shape(mu)[0]
    ln_det = jnp.sum(jnp.log(jnp.diag(L)))
    L_inv_y_minus_mu = jsp.linalg.solve_triangular(L, y - mu, lower=True)
    mahalanobis_squared = L_inv_y_minus_mu.T @ L_inv_y_minus_mu
    return -d * (jnp.log(2 * jnp.pi)) / 2 - ln_det - 0.5 * mahalanobis_squared

def jax_chi_factors(chi: ArrayLike, coeffs: ArrayLike):
    log1mc = jnp.log1p(-chi)
    log1mc2 = log1mc * log1mc
    log1mc3 = log1mc2 * log1mc
    log1mc4 = log1mc2 * log1mc2
    v = jnp.stack([chi, jnp.array(1.0), log1mc, log1mc2, log1mc3, log1mc4])
    return jnp.dot(coeffs, v)


def jax_a_from_quadratures(
    Apx: ArrayLike, Apy: ArrayLike, Acx: ArrayLike, Acy: ArrayLike
) -> Array:
    """Get the canonical amplitude from the quadratures

    Parameters
    ==========
    Apx : ArrayLike
        The amplitude of the "plus" cosine-like quadrature.
    Apy : ArrayLike
        The amplitude of the "plus" sine-like quadrature.
    Acx : ArrayLike
        The amplitude of the "cross" cosine-like quadrature.
    Acy : ArrayLike
        The amplitude of the "cross" sine-like quadrature.

    Returns
    =======
    Array
        The array of canonical amplitudes for each mode
    """
    # See arXiv:2208.03372 eq 47 & 57
    A = 0.5 * (
        jnp.sqrt(jnp.square(Acy + Apx) + jnp.square(Acx - Apy))
        + jnp.sqrt(jnp.square(Acy - Apx) + jnp.square(Acx + Apy))
    )
    return A


def jax_ellip_from_quadratures(
    Apx: ArrayLike, Apy: ArrayLike, Acx: ArrayLike, Acy: ArrayLike
) -> Array:
    """Get the canonical amplitude from the quadratures

    Parameters
    ==========
    Apx : ArrayLike
        The amplitude of the "plus" cosine-like quadrature.
    Apy : ArrayLike
        The amplitude of the "plus" sine-like quadrature.
    Acx : ArrayLike
        The amplitude of the "cross" cosine-like quadrature.
    Acy : ArrayLike
        The amplitude of the "cross" sine-like quadrature.

    Returns
    =======
    Array
        The array of canonical ellipticities for each mode
    """
    # See arXiv:2208.03372 eq 47 & 57
    A = a_from_quadratures(Apx, Apy, Acx, Acy)
    e = (
        0.5
        * (
            jnp.sqrt(jnp.square(Acy + Apx) + jnp.square(Acx - Apy))
            - jnp.sqrt(jnp.square(Acy - Apx) + jnp.square(Acx + Apy))
        )
        / A
    )
    return e


def jax_Aellip_from_quadratures(
    Apx: ArrayLike, Apy: ArrayLike, Acx: ArrayLike, Acy: ArrayLike
) -> Tuple[Array, Array]:
    """Get the canonical amplitude from the quadratures

    Parameters
    ==========
    Apx : ArrayLike
        The amplitude of the "plus" cosine-like quadrature.
    Apy : ArrayLike
        The amplitude of the "plus" sine-like quadrature.
    Acx : ArrayLike
        The amplitude of the "cross" cosine-like quadrature.
    Acy : ArrayLike
        The amplitude of the "cross" sine-like quadrature.

    Returns
    =======
    Array
        The array of canonical amplitudes for each mode
    Array
        The array of canonical ellipticities for each mode
    """
    # See arXiv:2208.03372 eq 47 & 57
    # should be slightly cheaper than calling the two functions separately
    term1 = jnp.sqrt(jnp.square(Acy + Apx) + jnp.square(Acx - Apy))
    term2 = jnp.sqrt(jnp.square(Acy - Apx) + jnp.square(Acx + Apy))
    A = 0.5 * (term1 + term2)
    e = 0.5 * (term1 - term2) / A
    return A, e


def jax_phiR_from_quadratures(
    Apx: ArrayLike, Apy: ArrayLike, Acx: ArrayLike, Acy: ArrayLike
) -> Array:
    """Get the right handed phase from the quadratures

    Parameters
    ==========
    Apx : ArrayLike
        The amplitude of the "plus" cosine-like quadrature.
    Apy : ArrayLike
        The amplitude of the "plus" sine-like quadrature.
    Acx : ArrayLike
        The amplitude of the "cross" cosine-like quadrature.
    Acy : ArrayLike
        The amplitude of the "cross" sine-like quadrature.

    Returns
    =======
    Array
        The array of right handed phases for each mode
    """
    # See arXiv:2208.03372 eq 47 & 57
    return jnp.arctan2(-Acx + Apy, Acy + Apx)


def jax_phiL_from_quadratures(
    Apx: ArrayLike, Apy: ArrayLike, Acx: ArrayLike, Acy: ArrayLike
) -> Array:
    """Get the left handed phase from the quadratures

    Parameters
    ==========
    Apx : ArrayLike
        The amplitude of the "plus" cosine-like quadrature.
    Apy : ArrayLike
        The amplitude of the "plus" sine-like quadrature.
    Acx : ArrayLike
        The amplitude of the "cross" cosine-like quadrature.
    Acy : ArrayLike
        The amplitude of the "cross" sine-like quadrature.

    Returns
    =======
    Array
        The array of left handed phases for each mode
    """
    # See arXiv:2208.03372 eq 47 & 57
    return jnp.arctan2(-Acx - Apy, -Acy + Apx)


def jax_flat_A_quadratures_log_prior(
    Apx: ArrayLike, Apy: ArrayLike, Acx: ArrayLike, Acy: ArrayLike, flat_A: ArrayLike
) -> Array:
    """Compute the log prior element to convert from normal in quadratures to flat in quadratures

    Parameters
    ==========
    Apx_unit : ArrayLike
        The amplitude of the "plus" cosine-like quadrature, before scaling.
    Apy_unit : ArrayLike
        The amplitude of the "plus" sine-like quadrature, before scaling.
    Acx_unit : ArrayLike
        The amplitude of the "cross" cosine-like quadrature, before scaling.
    Acy_unit : ArrayLike
        The amplitude of the "cross" sine-like quadrature, before scaling.
    flat_A : ArrayLike
        The array of T/F determining whether to do conversion for each mode

    Returns
    =======
    Array
        The log probability term to convert to flat in quadratures prior, for the modes requested
    """
    return jnp.where(
        flat_A, (Apx**2 + Apy**2 + Acx**2 + Acy**2)/2, 0
    ).sum()


def jax_flat_A_log_jacobian(A: ArrayLike, flat_A: ArrayLike) -> Array:
    """Compute the log jacobian for converting to flat in canonical amplitude prior

    Parameters
    ==========
    A : ArrayLike
        The canonical amplitude for each mode
    flat_A : ArrayLike
        The array of T/F determining whether to do conversion for each mode

    Returns
    =======
    Array
        The log jacobian for converting from flat in quadratures to flat in canonical amplitude
    """
    return jnp.where(flat_A, -3 * jnp.log(A), 0).sum()


def jax_flat_A_ellip_log_jacobian(
    A: ArrayLike, ellip: ArrayLike, flat_A_ellip: ArrayLike
) -> Array:
    """Compute the log jacobian for converting to a flat prior in canonical amplitude and ellipticity

    Parameters
    ==========
    A : ArrayLike
        The canonical amplitude for each mode
    ellip : ArrayLike
        The canonical ellipticity for each mode
    flat_A_ellip : ArrayLike
        The array of T/F determining whether to do conversion for each mode

    Returns
    =======
    Array
        The log jacobian for converting from flat in quadratures to flat in canonical amplitude and ellipticity
    """
    return jnp.where(flat_A_ellip, -3 * jnp.log(A) - jnp.log1p(-(ellip**2)), 0).sum()


def jax_rd(
    ts: ArrayLike,
    f: ArrayLike,
    gamma: ArrayLike,
    Apx: ArrayLike,
    Apy: ArrayLike,
    Acx: ArrayLike,
    Acy: ArrayLike,
    Fp: ArrayLike,
    Fc: ArrayLike,
):
    """Generate a ringdown waveform as it appears in a detector.
    Name temporary until more thorough replacement is ready.

    Parameters
    ==========
    ts : ArrayLike
        The times at which the ringdown waveform should be evaluated.
    f : ArrayLike
        The frequency of the ringdown mode.
    gamma : ArrayLike
        The damping rate.
    Apx : ArrayLike
        The amplitude of the "plus" cosine-like quadrature.
    Apy : ArrayLike
        The amplitude of the "plus" sine-like quadrature.
    Acx : ArrayLike
        The amplitude of the "cross" cosine-like quadrature.
    Acy : ArrayLike
        The amplitude of the "cross" sine-like quadrature.
    Fp : ArrayLike
        The coefficient of the "plus" polarization in the detector.
    Fc : ArrayLike
        The coefficient of the "cross" term in the detector.

    Returns
    =======
    Array
        Array of the ringdown waveform in the detector.
    """
    ct = jnp.cos(2 * jnp.pi * f * ts)
    st = jnp.sin(2 * jnp.pi * f * ts)
    decay = jnp.exp(-gamma * ts)
    p = decay * (Apx * ct + Apy * st)
    c = decay * (Acx * ct + Acy * st)
    return Fp * p + Fc * c


def jax_compute_h_det_mode(
    t0s: ArrayLike,
    ts: ArrayLike,
    Fps: ArrayLike,
    Fcs: ArrayLike,
    fs: ArrayLike,
    gammas: ArrayLike,
    Apxs: ArrayLike,
    Apys: ArrayLike,
    Acxs: ArrayLike,
    Acys: ArrayLike,
) -> Array:
    """Return the ringdown projected on the detector for a given mode

    Parameters
    ==========
    t0s : ArrayLike
        The starting time in each detector
    ts : ArrayLike
        The times at which the ringdown waveform should be evaluated in each detector.
    Fps : ArrayLike
        The coefficient of the "plus" polarization in each detector
    Fcs : ArrayLike
        The coefficient of the "cross" term in each detector
    fs : ArrayLike
        The frequencies of each mode.
    gammas : ArrayLike
        The damping rates of each mode.
    Apx : ArrayLike
        The amplitude of the "plus" cosine-like quadrature for each mode.
    Apy : ArrayLike
        The amplitude of the "plus" sine-like quadrature for each mode.
    Acx : ArrayLike
        The amplitude of the "cross" cosine-like quadrature for each mode.
    Acy : ArrayLike
        The amplitude of the "cross" sine-like quadrature for each mode.

    Returns
    =======
    Array
        The ringdown time series in each detector
    """
    ndet = len(t0s)
    nmode = fs.shape[0]
    nsamp = ts[0].shape[0]

    t0s = t0s.reshape((ndet, 1, 1))
    ts = ts.reshape((ndet, 1, nsamp))
    Fps = Fps.reshape((ndet, 1, 1))
    Fcs = Fcs.reshape((ndet, 1, 1))
    fs = fs.reshape((1, nmode, 1))
    gammas = gammas.reshape((1, nmode, 1))
    Apxs = Apxs.reshape((1, nmode, 1))
    Apys = Apys.reshape((1, nmode, 1))
    Acxs = Acxs.reshape((1, nmode, 1))
    Acys = Acys.reshape((1, nmode, 1))

    # Compare to arXiv:2107.05609 eq 13 & 14

    return jax_rd(ts - t0s, fs, gammas, Apxs, Apys, Acxs, Acys, Fps, Fcs)


def jax_make_mchi_model(
    t0: ArrayLike,
    times: ArrayLike,
    strains: ArrayLike,
    Ls: ArrayLike,
    Fps: ArrayLike,
    Fcs: ArrayLike,
    f_coeffs: ArrayLike,
    g_coeffs: ArrayLike,
    **kwargs,
) -> callable:
    """Make an appropriate M-chi model.
    In flowmc terms, the 'model' so to speak is the probability function logp.

    Parameters
    ==========

    """
    M_min = kwargs.pop("M_min")
    M_max = kwargs.pop("M_max")
    chi_min = kwargs.pop("chi_min")
    chi_max = kwargs.pop("chi_max")
    A_scale = kwargs.pop("A_scale")
    df_min = kwargs.pop("df_min")
    df_max = kwargs.pop("df_max")
    dtau_min = kwargs.pop("dtau_min")
    dtau_max = kwargs.pop("dtau_max")
    perturb_f = kwargs.pop("perturb_f", 0)
    perturb_tau = kwargs.pop("perturb_tau", 0)
    flat_A = kwargs.pop("flat_A", True)
    flat_A_ellip = kwargs.pop("flat_A_ellip", False)
    f_min = kwargs.pop("f_min", None)
    f_max = kwargs.pop("f_max", None)
    prior_run = kwargs.pop("prior_run", False)

    nmode = f_coeffs.shape[0]

    if np.isscalar(flat_A):
        flat_A = np.repeat(flat_A, nmode)
    if np.isscalar(flat_A_ellip):
        flat_A_ellip = np.repeat(flat_A_ellip, nmode)
    elif len(flat_A) != nmode:
        raise ValueError(
            "flat_A must either be a scalar or array of length equal to the number of modes"
        )
    elif len(flat_A_ellip) != nmode:
        raise ValueError(
            "flat_A_ellip must either be a scalar or array of length equal to the number of modes"
        )

    if any(flat_A) and any(flat_A_ellip):
        raise ValueError(
            "at most one of `flat_A` and `flat_A_ellip` can have an element that is "
            "`True`"
        )
    if (chi_min < 0) or (chi_max > 1):
        raise ValueError("chi boundaries must be contained in [0, 1)")

    if not np.isscalar(df_min) and not np.isscalar(df_max):
        if len(df_min) != len(df_max):
            raise ValueError(
                "df_min, df_max must be scalar or arrays of length equal to the number of modes"
            )
        for el in np.arange(len(df_min)):
            if df_min[el] == df_max[el]:
                raise ValueError(
                    "df_min and df_max must not be equal for any given mode"
                )

    if not np.isscalar(dtau_min) and not np.isscalar(dtau_max):
        if len(dtau_min) != len(dtau_max):
            raise ValueError(
                "dtau_min, dtau_max must be scalar or arrays of length equal to the number of modes"
            )
        for el in np.arange(len(dtau_min)):
            if dtau_min[el] == dtau_max[el]:
                raise ValueError(
                    "dtau_min and dtau_max must not be equal for any given mode"
                )

    ndet = len(t0)
    nt = len(times[0])

    ifos = kwargs.pop("ifos", np.arange(ndet))
    modes = kwargs.pop("modes", np.arange(nmode))

    # Use an if statement in the factory to generate the appropriate amplitude prior transform.
    # In reality this is probably not necessary, and jnp.where could handle everything but
    # Better safe than sorry
    # If XLAs compilation is worse than I think it is this may save a function call per loop
    if any(flat_A):

        def _amplitude_prior_transform(
            Apx_unit, Apy_unit, Acx_unit, Acy_unit, A, ellip, flat_A
        ):
            # Go to flat in quadratures
            flat_in_quadratures_log_transform = jax_flat_A_quadratures_log_prior(
                Apx_unit, Apy_unit, Acx_unit, Acy_unit, flat_A
            )
            # Then flat in A
            flat_in_A_log_jacobian = jnp.where(
                jnp.any(A) == 0, -jnp.inf, jax_flat_A_log_jacobian(A, flat_A)
            ).sum()

            return jnp.nan_to_num(
                flat_in_quadratures_log_transform + flat_in_A_log_jacobian
            )

    elif any(flat_A_ellip):

        def _amplitude_prior_transform(
            Apx_unit, Apy_unit, Acx_unit, Acy_unit, A, ellip, flat_A_ellip
        ):
            flat_in_quadratures_log_transform = jax_flat_A_quadratures_log_prior(
                Apx_unit, Apy_unit, Acx_unit, Acy_unit, flat_A_ellip
            )
            flat_in_A_ellip_log_jacobian = jnp.where(
                jnp.any(A) == 0, -jnp.inf, jax_flat_A_ellip_log_jacobian(A, ellip, flat_A_ellip)
            ).sum()
            return flat_in_quadratures_log_transform + flat_in_A_ellip_log_jacobian

    else:
        # If we're staying in normal in quadratures, just return 0 = log(1)
        _amplitude_prior_transform = (
            lambda Apx_unit, Apy_unit, Acx_unit, Acy_unit, A, ellip, flat_A: jnp.array(
                0
            )
        )

    @jit
    def mchi_logp(x: ArrayLike, *args, **kws) -> Array:
        """Get a model for log probability in an M-chi model

        Parameters
        ==========
        x : ArrayLike
            An array of all the parameters for the model, in the following order:
            x[0] = M : The mass of the remnant
            x[1] = chi : The spin of the remnant
            x[2 : 2 + nmode] = Apx_unit : The amplitude of the "plus" cosine-like quadrature, before scaling.
            x[2 + nmode : 2 + 2 * nmode] = Apy_unit : The amplitude of the "plus" sine-like quadrature, before scaling.
            x[2 + 2 * nmode : 2 + 3 * nmode] = Acx_unit : The amplitude of the "cross" cosine-like quadrature, before scaling.
            x[2 + 3 * nmode : 2 + 4 * nmode] = Acy_unit : The amplitude of the "cross" sine-like quadrature, before scaling.
            x[2 + 4 * nmode : 2 + 5 * nmode] = df : The frequency perturbation for each mode
            x[2 + 5 * nmode : 2 + 6 * nmode] = dtau : The damping time perturbation for each mode

        Returns
        =======
        Array
            The log posterior (log L + log pi) of this configuration
        """
        M = x[0]
        chi = x[1]
        Apx_unit = x[2 : 2 + nmode]
        Apy_unit = x[2 + nmode : 2 + 2 * nmode]
        Acx_unit = x[2 + 2 * nmode : 2 + 3 * nmode]
        Acy_unit = x[2 + 3 * nmode : 2 + 4 * nmode]
        df = x[2 + 4 * nmode : 2 + 5 * nmode]
        dtau = x[2 + 5 * nmode : 2 + 6 * nmode]

        # See arXiv:2208.03372 sec V.d for quadrature details
        Apx = jnp.multiply(Apx_unit, A_scale)
        Apy = jnp.multiply(Apy_unit, A_scale)
        Acx = jnp.multiply(Acx_unit, A_scale)
        Acy = jnp.multiply(Acy_unit, A_scale)

        A, ellip = jax_Aellip_from_quadratures(Apx, Apy, Acx, Acy)

        f0 = FREF * MREF / M
        fs = f0 * jax_chi_factors(chi, f_coeffs) * jnp.exp(df * perturb_f)
        gammas = f0 * jax_chi_factors(chi, g_coeffs) * jnp.exp(-dtau * perturb_tau)

        # Priors:

        # Flat in M, chi
        m_log_prior = jnp.where(
            (M <= M_max) & (M >= M_min), -jnp.log(M_max - M_min), -jnp.inf
        ).sum()
        chi_log_prior = jnp.where(
            (chi <= chi_max) & (chi >= chi_min), -jnp.log(chi_max - chi_min), -jnp.inf
        ).sum()

        # flat in f, tau
        df_log_prior = jnp.where(
            (df <= df_max) & (df >= df_min), -jnp.log(df_max - df_min), -jnp.inf
        ).sum()
        dtau_log_prior = jnp.where(
            (dtau <= dtau_max) & (dtau >= dtau_min),
            -jnp.log(dtau_max - dtau_min),
            -jnp.inf,
        ).sum()

        # Frequency cuts
        freq_cut_log_prior = jnp.where((fs < f_max) & (fs > f_min), 0, -jnp.inf).sum()

        # Amplitude prior
        # Normal in quadratures
        Apx_unit_log_prior = jsp.stats.multivariate_normal.logpdf(
            Apx_unit, mean=jnp.zeros(nmode), cov=jnp.identity(nmode)
        )
        Apy_unit_log_prior = jsp.stats.multivariate_normal.logpdf(
            Apy_unit, mean=jnp.zeros(nmode), cov=jnp.identity(nmode)
        )
        Acx_unit_log_prior = jsp.stats.multivariate_normal.logpdf(
            Acx_unit, mean=jnp.zeros(nmode), cov=jnp.identity(nmode)
        )
        Acy_unit_log_prior = jsp.stats.multivariate_normal.logpdf(
            Acy_unit, mean=jnp.zeros(nmode), cov=jnp.identity(nmode)
        )

        # flat_A | flat_A_ellip = flat_A xor flat_A_ellip, since one or the other is all False (per previous assertion)
        amplitude_transform_log_prior = _amplitude_prior_transform(
            Apx_unit, Apy_unit, Acx_unit, Acy_unit, A, ellip, flat_A | flat_A_ellip
        )

        # Sum it up to get the log prior
        log_prior = (
            m_log_prior
            + chi_log_prior
            + df_log_prior
            + dtau_log_prior
            + Apx_unit_log_prior
            + Apy_unit_log_prior
            + Acx_unit_log_prior
            + Acy_unit_log_prior
            + amplitude_transform_log_prior
            + freq_cut_log_prior
        )

        h_det_mode = jax_compute_h_det_mode(
            t0, times, Fps, Fcs, fs, gammas, Apx, Apy, Acx, Acy
        )
        h_det = jnp.sum(h_det_mode, axis=1)

        log_likelihoood = 0

        if not prior_run:
            for ii in range(ndet):
                log_likelihoood += jax_mvnorm_cholesky_logpdf(
                    strains[ii, :], h_det[ii, :], Ls[ii, :, :]
                )
        else:
            log_likelihoood += jnp.where(
                (A > 10 * A_scale) | (A > 1e-19), -jnp.inf, 0
            ).sum()


        return log_likelihoood + log_prior

    return mchi_logp


################################################################################
################################################################################
###########################    Old Implementation    ###########################
################################################################################
################################################################################


def _atl_cho_solve(L_and_lower, b):
    """Replacement for `aesara.tensor.slinalg.cho_solve` that enables backprop using two `solve_triangular`.

    Assumes `L` is lower triangular, and solves for `x` where :math:`L L^T x = b`.
    """
    L, lower = L_and_lower

    y = atl.solve_triangular(L, b, lower=lower)
    return atl.solve_triangular(L.T, y, lower=(not lower))


def rd(ts, f, gamma, Apx, Apy, Acx, Acy, Fp, Fc):
    """Generate a ringdown waveform as it appears in a detector.

    Arguments
    ---------

    ts : array_like
        The times at which the ringdown waveform should be evaluated.
    f : real
        The frequency.
    gamma : real
        The damping rate.
    Apx : real
        The amplitude of the "plus" cosine-like quadrature.
    Apy : real
        The amplitude of the "plus" sine-like quadrature.
    Acx : real
        The amplitude of the "cross" cosine-like quadrature.
    Acy : real
        The amplitude of the "cross" sine-like quadrature.
    Fp : real
        The coefficient of the "plus" polarization in the detector.
    Fc : real
        The coefficient of the "cross" term in the detector.

    Returns
    -------

    Array of the ringdown waveform in the detector.
    """
    ct = at.cos(2 * np.pi * f * ts)
    st = at.sin(2 * np.pi * f * ts)
    decay = at.exp(-gamma * ts)
    p = decay * (Apx * ct + Apy * st)
    c = decay * (Acx * ct + Acy * st)
    return Fp * p + Fc * c


def rd_design_matrix(t0s, ts, f, gamma, Fp, Fc, Ascales):
    ts = at.as_tensor(ts)
    nifo, nt = ts.shape

    nmode = f.shape[0]

    t0s = at.reshape(t0s, (nifo, 1, 1))
    ts = at.reshape(ts, (nifo, 1, nt))
    f = at.reshape(f, (1, nmode, 1))
    gamma = at.reshape(gamma, (1, nmode, 1))
    Fp = at.reshape(Fp, (nifo, 1, 1))
    Fc = at.reshape(Fc, (nifo, 1, 1))
    Ascales = at.reshape(Ascales, (1, nmode, 1))

    t = ts - t0s

    ct = at.cos(2 * np.pi * f * t)
    st = at.sin(2 * np.pi * f * t)
    decay = at.exp(-gamma * t)
    return at.concatenate(
        (
            Ascales * Fp * decay * ct,
            Ascales * Fp * decay * st,
            Ascales * Fc * decay * ct,
            Ascales * Fc * decay * st,
        ),
        axis=1,
    )


def chi_factors(chi, coeffs):
    log1mc = at.log1p(-chi)
    log1mc2 = log1mc * log1mc
    log1mc3 = log1mc2 * log1mc
    log1mc4 = log1mc2 * log1mc2
    v = at.stack([chi, at.as_tensor_variable(1.0), log1mc, log1mc2, log1mc3, log1mc4])
    return at.dot(coeffs, v)


def get_snr(h, d, L):
    wh = atl.solve_lower_triangular(L, h)
    wd = atl.solve_lower_triangular(L, h)
    return at.dot(wh, wd) / at.sqrt(at.dot(wh, wh))


def compute_h_det_mode(t0s, ts, Fps, Fcs, fs, gammas, Apxs, Apys, Acxs, Acys):
    ndet = len(t0s)
    nmode = fs.shape[0]
    nsamp = ts[0].shape[0]

    t0s = at.as_tensor_variable(t0s).reshape((ndet, 1, 1))
    ts = at.as_tensor_variable(ts).reshape((ndet, 1, nsamp))
    Fps = at.as_tensor_variable(Fps).reshape((ndet, 1, 1))
    Fcs = at.as_tensor_variable(Fcs).reshape((ndet, 1, 1))
    fs = at.as_tensor_variable(fs).reshape((1, nmode, 1))
    gammas = at.as_tensor_variable(gammas).reshape((1, nmode, 1))
    Apxs = at.as_tensor_variable(Apxs).reshape((1, nmode, 1))
    Apys = at.as_tensor_variable(Apys).reshape((1, nmode, 1))
    Acxs = at.as_tensor_variable(Acxs).reshape((1, nmode, 1))
    Acys = at.as_tensor_variable(Acys).reshape((1, nmode, 1))

    return rd(ts - t0s, fs, gammas, Apxs, Apys, Acxs, Acys, Fps, Fcs)


def a_from_quadratures(Apx, Apy, Acx, Acy):
    A = 0.5 * (
        at.sqrt(at.square(Acy + Apx) + at.square(Acx - Apy))
        + at.sqrt(at.square(Acy - Apx) + at.square(Acx + Apy))
    )
    return A


def ellip_from_quadratures(Apx, Apy, Acx, Acy):
    A = a_from_quadratures(Apx, Apy, Acx, Acy)
    e = (
        0.5
        * (
            at.sqrt(at.square(Acy + Apx) + at.square(Acx - Apy))
            - at.sqrt(at.square(Acy - Apx) + at.square(Acx + Apy))
        )
        / A
    )
    return e


def Aellip_from_quadratures(Apx, Apy, Acx, Acy):
    # should be slightly cheaper than calling the two functions separately
    term1 = at.sqrt(at.square(Acy + Apx) + at.square(Acx - Apy))
    term2 = at.sqrt(at.square(Acy - Apx) + at.square(Acx + Apy))
    A = 0.5 * (term1 + term2)
    e = 0.5 * (term1 - term2) / A
    return A, e


def phiR_from_quadratures(Apx, Apy, Acx, Acy):
    return at.arctan2(-Acx + Apy, Acy + Apx)


def phiL_from_quadratures(Apx, Apy, Acx, Acy):
    return at.arctan2(-Acx - Apy, -Acy + Apx)


def flat_A_quadratures_prior(Apx_unit, Apy_unit, Acx_unit, Acy_unit, flat_A):
    return 0.5 * at.sum(
        (
            at.square(Apx_unit)
            + at.square(Apy_unit)
            + at.square(Acx_unit)
            + at.square(Acy_unit)
        )
        * flat_A
    )


def make_mchi_model(t0, times, strains, Ls, Fps, Fcs, f_coeffs, g_coeffs, **kwargs):
    M_min = kwargs.pop("M_min")
    M_max = kwargs.pop("M_max")
    chi_min = kwargs.pop("chi_min")
    chi_max = kwargs.pop("chi_max")
    A_scale = kwargs.pop("A_scale")
    df_min = kwargs.pop("df_min")
    df_max = kwargs.pop("df_max")
    dtau_min = kwargs.pop("dtau_min")
    dtau_max = kwargs.pop("dtau_max")
    perturb_f = kwargs.pop("perturb_f", 0)
    perturb_tau = kwargs.pop("perturb_tau", 0)
    flat_A = kwargs.pop("flat_A", True)
    flat_A_ellip = kwargs.pop("flat_A_ellip", False)
    f_min = kwargs.pop("f_min", None)
    f_max = kwargs.pop("f_max", None)
    prior_run = kwargs.pop("prior_run", False)

    nmode = f_coeffs.shape[0]

    if np.isscalar(flat_A):
        flat_A = np.repeat(flat_A, nmode)
    if np.isscalar(flat_A_ellip):
        flat_A_ellip = np.repeat(flat_A_ellip, nmode)
    elif len(flat_A) != nmode:
        raise ValueError(
            "flat_A must either be a scalar or array of length equal to the number of modes"
        )
    elif len(flat_A_ellip) != nmode:
        raise ValueError(
            "flat_A_ellip must either be a scalar or array of length equal to the number of modes"
        )

    if any(flat_A) and any(flat_A_ellip):
        raise ValueError(
            "at most one of `flat_A` and `flat_A_ellip` can have an element that is "
            "`True`"
        )
    if (chi_min < 0) or (chi_max > 1):
        raise ValueError("chi boundaries must be contained in [0, 1)")
    ndet = len(t0)
    nt = len(times[0])

    ifos = kwargs.pop("ifos", np.arange(ndet))
    modes = kwargs.pop("modes", np.arange(nmode))

    coords = {"ifo": ifos, "mode": modes, "time_index": np.arange(nt)}

    with pm.Model(coords=coords) as model:
        pm.ConstantData("times", times, dims=["ifo", "time_index"])
        pm.ConstantData("t0", t0, dims=["ifo"])
        pm.ConstantData("L", Ls, dims=["ifo", "time_index", "time_index"])

        M = pm.Uniform("M", M_min, M_max)
        chi = pm.Uniform("chi", chi_min, chi_max)

        Apx_unit = pm.Normal("Apx_unit", dims=["mode"])
        Apy_unit = pm.Normal("Apy_unit", dims=["mode"])
        Acx_unit = pm.Normal("Acx_unit", dims=["mode"])
        Acy_unit = pm.Normal("Acy_unit", dims=["mode"])

        df = pm.Uniform("df", df_min, df_max, dims=["mode"])
        dtau = pm.Uniform("dtau", dtau_min, dtau_max, dims=["mode"])

        Apx = pm.Deterministic("Apx", A_scale * Apx_unit, dims=["mode"])
        Apy = pm.Deterministic("Apy", A_scale * Apy_unit, dims=["mode"])
        Acx = pm.Deterministic("Acx", A_scale * Acx_unit, dims=["mode"])
        Acy = pm.Deterministic("Acy", A_scale * Acy_unit, dims=["mode"])

        A = pm.Deterministic("A", a_from_quadratures(Apx, Apy, Acx, Acy), dims=["mode"])
        ellip = pm.Deterministic(
            "ellip", ellip_from_quadratures(Apx, Apy, Acx, Acy), dims=["mode"]
        )

        f0 = FREF * MREF / M
        f = pm.Deterministic(
            "f", f0 * chi_factors(chi, f_coeffs) * at.exp(df * perturb_f), dims=["mode"]
        )
        gamma = pm.Deterministic(
            "gamma",
            f0 * chi_factors(chi, g_coeffs) * at.exp(-dtau * perturb_tau),
            dims=["mode"],
        )
        tau = pm.Deterministic("tau", 1 / gamma, dims=["mode"])
        Q = pm.Deterministic("Q", np.pi * f * tau, dims=["mode"])
        phiR = pm.Deterministic(
            "phiR", phiR_from_quadratures(Apx, Apy, Acx, Acy), dims=["mode"]
        )
        phiL = pm.Deterministic(
            "phiL", phiL_from_quadratures(Apx, Apy, Acx, Acy), dims=["mode"]
        )
        theta = pm.Deterministic("theta", -0.5 * (phiR + phiL), dims=["mode"])
        phi = pm.Deterministic("phi", 0.5 * (phiR - phiL), dims=["mode"])

        # Check limits on f
        if not np.isscalar(f_min) or not f_min == 0.0:
            _ = pm.Potential("f_min_cut", at.sum(at.where(f < f_min, np.NINF, 0.0)))
        if not np.isscalar(f_max) or not f_max == np.inf:
            _ = pm.Potential("f_max_cut", at.sum(at.where(f > f_max, np.NINF, 0.0)))

        h_det_mode = pm.Deterministic(
            "h_det_mode",
            compute_h_det_mode(t0, times, Fps, Fcs, f, gamma, Apx, Apy, Acx, Acy),
            dims=["ifo", "mode", "time_index"],
        )
        h_det = pm.Deterministic(
            "h_det", at.sum(h_det_mode, axis=1), dims=["ifo", "time_index"]
        )

        # Priors:

        # Flat in M-chi already

        # Amplitude prior
        if any(flat_A):
            # bring us back to flat-in-quadratures
            pm.Potential(
                "flat_A_quadratures_prior",
                flat_A_quadratures_prior(
                    Apx_unit, Apy_unit, Acx_unit, Acy_unit, flat_A
                ),
            )
            # bring us to flat-in-A prior
            pm.Potential("flat_A_prior", -3 * at.sum(at.log(A) * flat_A))
        elif any(flat_A_ellip):
            # bring us back to flat-in-quadratures
            pm.Potential(
                "flat_A_quadratures_prior",
                flat_A_quadratures_prior(
                    Apx_unit, Apy_unit, Acx_unit, Acy_unit, flat_A_ellip
                ),
            )
            # bring us to flat-in-A and flat-in-ellip prior
            pm.Potential(
                "flat_A_ellip_prior",
                at.sum((-3 * at.log(A) - at.log1m(at.square(ellip))) * flat_A_ellip),
            )

        # Flat prior on the delta-fs and delta-taus

        # Likelihood:
        if not prior_run:
            for i in range(ndet):
                key = ifos[i]
                if isinstance(key, bytes):
                    # Don't want byte strings in our names!
                    key = key.decode("utf-8")
                _ = pm.MvNormal(
                    f"strain_{key}",
                    mu=h_det[i, :],
                    chol=Ls[i],
                    observed=strains[i],
                    dims=["time_index"],
                )
        else:
            print("Sampling prior")
            samp_prior_cond = pm.Potential(
                "A_prior", at.sum(at.where(A > (10 * A_scale or 1e-19), np.NINF, 0.0))
            )  # this condition is to bound flat priors just for sampling from the prior

        # This *should* be the log likelihood as we normally think of it
        log_likelihood = pm.Deterministic("log_likelihood", model.observedlogp)
        # These two terms should, taken together, form the log prior as we would think of it
        log_prior = pm.Deterministic("log_p_unobserved", model.varlogp_nojac)
        log_potential = pm.Deterministic("log_p_potential", model.potentiallogp)

        # Finally, all of them together should be the total log probability
        log_prob = pm.Deterministic("log_probability", log_likelihood + log_prior + log_potential)

        return model


def make_mchi_aligned_model(
    t0, times, strains, Ls, Fps, Fcs, f_coeffs, g_coeffs, **kwargs
):
    M_min = kwargs.pop("M_min")
    M_max = kwargs.pop("M_max")
    chi_min = kwargs.pop("chi_min")
    chi_max = kwargs.pop("chi_max")
    cosi_min = kwargs.pop("cosi_min")
    cosi_max = kwargs.pop("cosi_max")
    A_scale = kwargs.pop("A_scale")
    df_min = kwargs.pop("df_min")
    df_max = kwargs.pop("df_max")
    dtau_min = kwargs.pop("dtau_min")
    dtau_max = kwargs.pop("dtau_max")
    perturb_f = kwargs.pop("perturb_f", 0)
    perturb_tau = kwargs.pop("perturb_tau", 0)
    flat_A = kwargs.pop("flat_A", True)
    f_min = kwargs.pop("f_min", 0.0)
    f_max = kwargs.pop("f_max", np.inf)
    nmode = f_coeffs.shape[0]
    prior_run = kwargs.pop("prior_run", False)

    if np.isscalar(flat_A):
        flat_A = np.repeat(flat_A, nmode)
    elif len(flat_A) != nmode:
        raise ValueError(
            "flat_A must either be a scalar or array of length equal to the number of modes"
        )

    if (cosi_min < -1) or (cosi_max > 1):
        raise ValueError("cosi boundaries must be contained in [-1, 1]")
    if (chi_min < 0) or (chi_max > 1):
        raise ValueError("chi boundaries must be contained in [0, 1)")

    ndet = len(t0)
    nt = len(times[0])

    ifos = kwargs.pop("ifos", np.arange(ndet))
    modes = kwargs.pop("modes", np.arange(nmode))

    coords = {"ifo": ifos, "mode": modes, "time_index": np.arange(nt)}

    with pm.Model(coords=coords) as model:
        pm.ConstantData("times", times, dims=["ifo", "time_index"])
        pm.ConstantData("t0", t0, dims=["ifo"])
        pm.ConstantData("L", Ls, dims=["ifo", "time_index", "time_index"])

        M = pm.Uniform("M", M_min, M_max)
        chi = pm.Uniform("chi", chi_min, chi_max)

        cosi = pm.Uniform("cosi", cosi_min, cosi_max)

        Ax_unit = pm.Normal("Ax_unit", dims=["mode"])
        Ay_unit = pm.Normal("Ay_unit", dims=["mode"])

        df = pm.Uniform("df", df_min, df_max, dims=["mode"])
        dtau = pm.Uniform("dtau", dtau_min, dtau_max, dims=["mode"])

        A = pm.Deterministic(
            "A",
            A_scale * at.sqrt(at.square(Ax_unit) + at.square(Ay_unit)),
            dims=["mode"],
        )
        phi = pm.Deterministic("phi", at.arctan2(Ay_unit, Ax_unit), dims=["mode"])

        f0 = FREF * MREF / M
        f = pm.Deterministic(
            "f", f0 * chi_factors(chi, f_coeffs) * at.exp(df * perturb_f), dims=["mode"]
        )
        gamma = pm.Deterministic(
            "gamma",
            f0 * chi_factors(chi, g_coeffs) * at.exp(-dtau * perturb_tau),
            dims=["mode"],
        )
        tau = pm.Deterministic("tau", 1 / gamma, dims=["mode"])
        Q = pm.Deterministic("Q", np.pi * f * tau, dims=["mode"])
        Ap = pm.Deterministic("Ap", (1 + at.square(cosi)) * A, dims=["mode"])
        Ac = pm.Deterministic("Ac", 2 * cosi * A, dims=["mode"])
        ellip = pm.Deterministic("ellip", Ac / Ap, dims=["mode"])

        # Check limits on f
        if not np.isscalar(f_min) or not f_min == 0.0:
            _ = pm.Potential("f_min_cut", at.sum(at.where(f < f_min, np.NINF, 0.0)))
            print("Running with f_min_cut on modes:", f_min)
        if not np.isscalar(f_max) or not f_max == np.inf:
            _ = pm.Potential("f_max_cut", at.sum(at.where(f > f_max, np.NINF, 0.0)))
            print("Running with f_max_cut on modes:", f_max)

        Apx = (1 + at.square(cosi)) * A * at.cos(phi)
        Apy = (1 + at.square(cosi)) * A * at.sin(phi)
        Acx = -2 * cosi * A * at.sin(phi)
        Acy = 2 * cosi * A * at.cos(phi)

        h_det_mode = pm.Deterministic(
            "h_det_mode",
            compute_h_det_mode(t0, times, Fps, Fcs, f, gamma, Apx, Apy, Acx, Acy),
            dims=["ifo", "mode", "time_index"],
        )
        h_det = pm.Deterministic(
            "h_det", at.sum(h_det_mode, axis=1), dims=["ifo", "time_index"]
        )

        # Priors:

        # Flat in M-chi already

        # Amplitude prior
        if any(flat_A):
            # first bring us to flat in quadratures
            pm.Potential(
                "flat_A_quadratures_prior",
                0.5 * at.sum((at.square(Ax_unit) + at.square(Ay_unit)) * flat_A),
            )
            # now to flat in A
            pm.Potential("flat_A_prior", -at.sum(at.log(A) * flat_A))

        # Flat prior on the delta-fs and delta-taus

        # Likelihood
        if not prior_run:
            for i in range(ndet):
                key = ifos[i]
                if isinstance(key, bytes):
                    # Don't want byte strings in our names!
                    key = key.decode("utf-8")
                _ = pm.MvNormal(
                    f"strain_{key}",
                    mu=h_det[i, :],
                    chol=Ls[i],
                    observed=strains[i],
                    dims=["time_index"],
                )
        else:
            print("Sampling prior")
            samp_prior_cond = pm.Potential(
                "A_prior", at.sum(at.where(A > (10 * A_scale or 1e-19), np.NINF, 0.0))
            )  # this condition is to bound flat priors just for sampling from the prior

        return model


def make_mchi_marginalized_model(
    t0, times, strains, Ls, Fps, Fcs, f_coeffs, g_coeffs, **kwargs
):
    M_min = kwargs.pop("M_min")
    M_max = kwargs.pop("M_max")
    chi_min = kwargs.pop("chi_min")
    chi_max = kwargs.pop("chi_max")
    A_scale_max = kwargs.pop("A_scale_max")
    df_min = kwargs.pop("df_min")
    df_max = kwargs.pop("df_max")
    dtau_min = kwargs.pop("dtau_min")
    dtau_max = kwargs.pop("dtau_max")
    perturb_f = kwargs.pop("perturb_f", 0)
    perturb_tau = kwargs.pop("perturb_tau", 0)
    f_min = kwargs.pop("f_min", None)
    f_max = kwargs.pop("f_max", None)
    prior_run = kwargs.pop("prior_run", False)

    nmode = f_coeffs.shape[0]

    if (chi_min < 0) or (chi_max > 1):
        raise ValueError("chi boundaries must be contained in [0, 1)")

    if not np.isscalar(df_min) and not np.isscalar(df_max):
        if len(df_min) != len(df_max):
            raise ValueError(
                "df_min, df_max must be scalar or arrays of length equal to the number of modes"
            )
        for el in np.arange(len(df_min)):
            if df_min[el] == df_max[el]:
                raise ValueError(
                    "df_min and df_max must not be equal for any given mode"
                )

    if not np.isscalar(dtau_min) and not np.isscalar(dtau_max):
        if len(dtau_min) != len(dtau_max):
            raise ValueError(
                "dtau_min, dtau_max must be scalar or arrays of length equal to the number of modes"
            )
        for el in np.arange(len(dtau_min)):
            if dtau_min[el] == dtau_max[el]:
                raise ValueError(
                    "dtau_min and dtau_max must not be equal for any given mode"
                )

    ndet = len(t0)
    nt = len(times[0])

    ifos = kwargs.pop("ifos", np.arange(ndet))
    modes = kwargs.pop("modes", np.arange(nmode))

    coords = {"ifo": ifos, "mode": modes, "time_index": np.arange(nt)}

    Llogdet = np.array([np.sum(np.log(np.diag(L))) for L in Ls])

    with pm.Model(coords=coords) as model:
        pm.ConstantData("times", times, dims=["ifo", "time_index"])
        pm.ConstantData("t0", t0, dims=["ifo"])
        pm.ConstantData("L", Ls, dims=["ifo", "time_index", "time_index"])

        M = pm.Uniform("M", M_min, M_max)
        chi = pm.Uniform("chi", chi_min, chi_max)

        df = pm.Uniform("df", df_min, df_max, dims=["mode"])
        dtau = pm.Uniform("dtau", dtau_min, dtau_max, dims=["mode"])

        # log_A_scale = pm.Uniform('log_A_scale', np.log(A_scale_min), np.log(A_scale_max), dims=['mode'])
        # A_scale = pm.Deterministic('A_scale', at.exp(log_A_scale))
        A_scale = pm.Uniform("A_scale", 0, A_scale_max, dims=["mode"])

        Apx_unit = pm.Normal("Apx_unit", dims=["mode"])
        Apy_unit = pm.Normal("Apy_unit", dims=["mode"])
        Acx_unit = pm.Normal("Acx_unit", dims=["mode"])
        Acy_unit = pm.Normal("Acy_unit", dims=["mode"])

        f0 = FREF * MREF / M
        f = pm.Deterministic(
            "f", f0 * chi_factors(chi, f_coeffs) * at.exp(df * perturb_f), dims=["mode"]
        )
        gamma = pm.Deterministic(
            "gamma",
            f0 * chi_factors(chi, g_coeffs) * at.exp(-dtau * perturb_tau),
            dims=["mode"],
        )
        tau = pm.Deterministic("tau", 1 / gamma, dims=["mode"])
        Q = pm.Deterministic("Q", np.pi * f * tau, dims=["mode"])

        # Check limits on f
        if not np.isscalar(f_min) or not f_min == 0.0:
            _ = pm.Potential("f_min_cut", at.sum(at.where(f < f_min, np.NINF, 0.0)))
        if not np.isscalar(f_max) or not f_max == np.inf:
            _ = pm.Potential("f_max_cut", at.sum(at.where(f > f_max, np.NINF, 0.0)))

        # Priors:

        # Flat in M-chi already

        # Flat prior on the delta-fs and delta-taus

        design_matrices = rd_design_matrix(t0, times, f, gamma, Fps, Fcs, A_scale)

        mu = at.zeros(4 * nmode)
        Lambda_inv = at.eye(4 * nmode)
        Lambda_inv_chol = at.eye(4 * nmode)
        # Likelihood:
        if not prior_run:
            for i in range(ndet):
                MM = design_matrices[
                    i, :, :
                ].T  # (ndet, 4*nmode, ntime) => (i, ntime, 4*nmode)

                A_inv = Lambda_inv + at.dot(MM.T, _atl_cho_solve((Ls[i], True), MM))
                A_inv_chol = atl.cholesky(A_inv)

                a = _atl_cho_solve(
                    (A_inv_chol, True),
                    at.dot(Lambda_inv, mu)
                    + at.dot(MM.T, _atl_cho_solve((Ls[i], True), strains[i])),
                )

                b = at.dot(MM, mu)

                Blogsqrtdet = (
                    Llogdet[i]
                    - at.sum(at.log(at.diag(Lambda_inv_chol)))
                    + at.sum(at.log(at.diag(A_inv_chol)))
                )

                r = strains[i] - b
                Cinv_r = _atl_cho_solve((Ls[i], True), r)
                MAMTCinv_r = at.dot(
                    MM, _atl_cho_solve((A_inv_chol, True), at.dot(MM.T, Cinv_r))
                )
                CinvMAMTCinv_r = _atl_cho_solve((Ls[i], True), MAMTCinv_r)
                logl = -0.5 * at.dot(r, Cinv_r - CinvMAMTCinv_r) - Blogsqrtdet

                key = ifos[i]
                if isinstance(key, bytes):
                    # Don't want byte strings in our names!
                    key = key.decode("utf-8")

                pm.Potential(f"strain_{key}", logl)

                mu = a
                Lambda_inv = A_inv
                Lambda_inv_chol = A_inv_chol
        else:
            # We're done.  There is no likelihood.
            pass

        # Lambda_inv_chol.T: Lambda_inv = Lambda_inv_chol * Lambda_inv_chol.T,
        # so Lambda = (Lambda_inv_chol.T)^{-1} Lambda_inv_chol^{-1} To achieve
        # the desired covariance, we can *right multiply* iid N(0,1) variables
        # by Lambda_inv_chol^{-1}, so that y = x Lambda_inv_chol^{-1} has
        # covariance < y^T y > = (Lambda_inv_chol^{-1}).T < x^T x >
        # Lambda_inv_chol^{-1} = (Lambda_inv_chol^{-1}).T I Lambda_inv_chol^{-1}
        # = Lambda.
        theta = mu + atl.solve(
            Lambda_inv_chol.T, at.concatenate((Apx_unit, Apy_unit, Acx_unit, Acy_unit))
        )

        Apx = pm.Deterministic("Apx", theta[:nmode] * A_scale, dims=["mode"])
        Apy = pm.Deterministic("Apy", theta[nmode : 2 * nmode] * A_scale, dims=["mode"])
        Acx = pm.Deterministic(
            "Acx", theta[2 * nmode : 3 * nmode] * A_scale, dims=["mode"]
        )
        Acy = pm.Deterministic("Acy", theta[3 * nmode :] * A_scale, dims=["mode"])

        A = pm.Deterministic("A", a_from_quadratures(Apx, Apy, Acx, Acy), dims=["mode"])
        ellip = pm.Deterministic(
            "ellip", ellip_from_quadratures(Apx, Apy, Acx, Acy), dims=["mode"]
        )

        phiR = pm.Deterministic(
            "phiR", phiR_from_quadratures(Apx, Apy, Acx, Acy), dims=["mode"]
        )
        phiL = pm.Deterministic(
            "phiL", phiL_from_quadratures(Apx, Apy, Acx, Acy), dims=["mode"]
        )
        theta = pm.Deterministic("theta", -0.5 * (phiR + phiL), dims=["mode"])
        phi = pm.Deterministic("phi", 0.5 * (phiR - phiL), dims=["mode"])

        h_det_mode = pm.Deterministic(
            "h_det_mode",
            compute_h_det_mode(t0, times, Fps, Fcs, f, gamma, Apx, Apy, Acx, Acy),
            dims=["ifo", "mode", "time_index"],
        )
        h_det = pm.Deterministic(
            "h_det", at.sum(h_det_mode, axis=1), dims=["ifo", "time_index"]
        )

        return model


def logit(p):
    return np.log(p) - np.log1p(-p)


def make_ftau_model(t0, times, strains, Ls, **kwargs):
    f_min = kwargs.pop("f_min")
    f_max = kwargs.pop("f_max")
    gamma_min = kwargs.pop("gamma_min")
    gamma_max = kwargs.pop("gamma_max")
    A_scale = kwargs.pop("A_scale")
    flat_A = kwargs.pop("flat_A", True)
    nmode = kwargs.pop("nmode", 1)
    prior_run = kwargs.pop("prior_run", False)

    if np.isscalar(flat_A):
        flat_A = np.repeat(flat_A, nmode)
    elif len(flat_A) != nmode:
        raise ValueError(
            "flat_A must either be a scalar or array of length equal to the number of modes"
        )

    ndet = len(t0)
    nt = len(times[0])

    ifos = kwargs.pop("ifos", np.arange(ndet))
    modes = kwargs.pop("modes", np.arange(nmode))

    coords = {"ifo": ifos, "mode": modes, "time_index": np.arange(nt)}

    with pm.Model(coords=coords) as model:
        pm.ConstantData("times", times, dims=["ifo", "time_index"])
        pm.ConstantData("t0", t0, dims=["ifo"])
        pm.ConstantData("L", Ls, dims=["ifo", "time_index", "time_index"])

        f = pm.Uniform("f", f_min, f_max, dims=["mode"])
        gamma = pm.Uniform(
            "gamma",
            gamma_min,
            gamma_max,
            dims=["mode"],
            transform=pm.distributions.transforms.multivariate_ordered,
        )

        Ax_unit = pm.Normal("Ax_unit", dims=["mode"])
        Ay_unit = pm.Normal("Ay_unit", dims=["mode"])

        A = pm.Deterministic(
            "A",
            A_scale * at.sqrt(at.square(Ax_unit) + at.square(Ay_unit)),
            dims=["mode"],
        )
        phi = pm.Deterministic("phi", at.arctan2(Ay_unit, Ax_unit), dims=["mode"])

        tau = pm.Deterministic("tau", 1 / gamma, dims=["mode"])
        Q = pm.Deterministic("Q", np.pi * f * tau, dims=["mode"])

        Apx = A * at.cos(phi)
        Apy = A * at.sin(phi)

        h_det_mode = pm.Deterministic(
            "h_det_mode",
            compute_h_det_mode(
                t0,
                times,
                np.ones(ndet),
                np.zeros(ndet),
                f,
                gamma,
                Apx,
                Apy,
                np.zeros(nmode),
                np.zeros(nmode),
            ),
            dims=["ifo", "mode", "time_index"],
        )
        h_det = pm.Deterministic(
            "h_det", at.sum(h_det_mode, axis=1), dims=["ifo", "time_index"]
        )

        # Priors:

        # Flat in M-chi already

        # Amplitude prior
        if any(flat_A):
            # first bring us to flat in quadratures
            pm.Potential(
                "flat_A_quadratures_prior",
                0.5 * at.sum((at.square(Ax_unit) + at.square(Ay_unit)) * flat_A),
            )
            pm.Potential("flat_A_prior", -at.sum(at.log(A) * flat_A))

        # Flat prior on the delta-fs and delta-taus

        # Likelihood
        if not prior_run:
            for i in range(ndet):
                key = ifos[i]
                if isinstance(key, bytes):
                    # Don't want byte strings in our names!
                    key = key.decode("utf-8")
                _ = pm.MvNormal(
                    f"strain_{key}",
                    mu=h_det[i, :],
                    chol=Ls[i],
                    observed=strains[i],
                    dims=["time_index"],
                )
        else:
            print("Sampling prior")
            samp_prior_cond = pm.Potential(
                "A_prior", at.sum(at.where(A > (10 * A_scale or 1e-19), np.NINF, 0.0))
            )  # this condition is to bound flat priors just for sampling from the prior

        return model
