import os

from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.backends import HDFBackend
from eryn.prior import uniform_dist, ProbDistContainer
import numpy as np
from eryn.utils import Update
from eryn.moves import StretchMove
from lisatools.sampling.moves.skymodehop import SkyMove
from bbhx.utils.transform import LISA_to_SSB, SSB_to_LISA

from pyHetWrap import pyHetLikeWrap

from scipy import constants


class HetUpdate:
    def __init__(self, het_like):
        self.het_like = het_like

    def __call__(self, iter, last_sample, sampler):
        best = np.where(last_sample.log_like == last_sample.log_like.max())
        best_params = last_sample.branches["mbh"].coords[best].squeeze().copy()

        # convert sky frame
        best_params[np.array([5, 8, 7, 9])] = convert_sky_coords(
            *best_params[np.array([5, 8, 7, 9])], convert_lisa_to_ssb=True
        )

        # convert beta-> theta
        best_params[7] = np.cos(np.pi / 2.0 - np.arcsin(best_params[7]))

        # update the reference waveform
        self.het_like.udpate_heterodyne(best_params)

        last_sample.log_like[0, 0] = like_func_wrap(
            last_sample.branches["mbh"].coords[0, 0, 0], self.het_like
        )


def convert_sky_coords(t_old, phi_old, sinbeta_old, psi_old, convert_lisa_to_ssb=True):
    lam_old = phi_old
    beta_old = np.arcsin(sinbeta_old)

    if convert_lisa_to_ssb:
        func = LISA_to_SSB
    else:
        func = SSB_to_LISA

    t_new, lam_new, beta_new, psi_new = func(t_old, lam_old, beta_old, psi_old)

    phi_new = lam_new
    sinbeta_new = np.sin(beta_new)

    phi_new = phi_new % (2 * np.pi)
    psi_new = psi_new % (np.pi)

    return (t_new, phi_new, sinbeta_new, psi_new)


def like_func_wrap(x, like_fn):
    x[np.array([5, 8, 7, 9])] = convert_sky_coords(
        *x[np.array([5, 8, 7, 9])], convert_lisa_to_ssb=True
    )

    # convert beta -> theta
    x[7] = np.cos(np.pi / 2.0 - np.arcsin(x[7]))
    try:
        out = like_fn.get_like(x)
    except ValueError:
        breakpoint()

    if np.isnan(out):
        out = -1e300

    return out


if __name__ == "__main__":
    Tobs = 1.0 * constants.Julian_year
    dt = 10.0

    ndims = {"mbh": 11}
    nwalkers = 100
    ntemps = 10

    Tstart = 0.0
    Tend = Tobs

    priors_in = {
        0: uniform_dist(np.log(1.0e2), np.log(0.44 * 5.0e8)),
        1: uniform_dist(np.log(1.0e3), np.log(5.0e8)),
        2: uniform_dist(-0.999, 0.999),
        3: uniform_dist(-0.999, 0.999),
        4: uniform_dist(0.0, np.pi),
        5: uniform_dist(1.01 * Tstart, 2 * Tend),
        6: uniform_dist(np.log(0.1), np.log(1.0e3)),
        7: uniform_dist(-1.0, 1.0),
        8: uniform_dist(0.0, 2 * np.pi),
        9: uniform_dist(0.0, 2 * np.pi),
        10: uniform_dist(-1.0, 1.0),
    }

    priors = {"mbh": ProbDistContainer(priors_in)}

    # Mc = 0.2e6
    # Mt = 1e6
    # a1 = 0.5
    # a2 = 0.5
    # phi_ref = 2.0
    # t_ref = 1e7
    # dist = 10.0  # Gpc
    # sinbeta = 0.6
    # phi = 2.5
    # psi = 0.7
    # cosinc = 0.2

    premove = np.genfromtxt("search_sources.dat")[:, 2:]
    NS = len(premove)

    m1, m2 = premove[:, :2].T.copy()
    premove[:, 0] = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
    premove[:, 1] = m1 + m2

    #  premove[:, 4] = (premove[:, 4] + 1 / 2 * np.pi / 2.0) % (2 * np.pi)
    premove[:, 0] = np.log(premove[:, 0])
    premove[:, 1] = np.log(premove[:, 1])
    premove[:, 6] = np.log(premove[:, 6])
    premove[:, 7] = np.sin(np.pi / 2.0 - np.arccos(premove[:, 7]))

    is_noise_free = False
    seg = 0
    rep = 0

    injection_params = premove[rep].copy()
    premove_in = premove.flatten().copy()

    # injection_params = np.array(
    #     [
    #         np.log(Mc),
    #         np.log(Mt),
    #         a1,
    #         a2,
    #         phi_ref,
    #         t_ref,
    #         np.log(dist),
    #         sinbeta,
    #         phi,
    #         psi,
    #         cosinc,
    #     ]
    # )

    moves = [
        (SkyMove(which="both", ind_map=dict(cosinc=10, sinbeta=7, lam=8, psi=9)), 0.02),
        (SkyMove(which="long", ind_map=dict(cosinc=10, sinbeta=7, lam=8, psi=9)), 0.05),
        (SkyMove(which="lat", ind_map=dict(cosinc=10, sinbeta=7, lam=8, psi=9)), 0.05),
        (StretchMove(), 0.88),
    ]

    like_fn = pyHetLikeWrap(
        injection_params, Tobs, dt, is_noise_free, seg, rep, premove_in, NS
    )

    breakpoint()
    file_store = "mbh_test.h5"

    if file_store in os.listdir():
        # start from file i.e. continue run
        reader = HDFBackend(file_store)
        start_state = reader.get_last_sample()

    else:
        # need to map start parameters to LISA frame
        injection_params_L_frame = injection_params.copy()
        injection_params_L_frame[np.array([5, 8, 7, 9])] = convert_sky_coords(
            *injection_params[np.array([5, 8, 7, 9])], convert_lisa_to_ssb=False
        )

        # start around true point
        tmp_cov = np.ones_like(injection_params_L_frame) * 1e-3

        # can adjust relative values if you wnat
        # tmp_cov[0] = .....

        factor = 1e-7

        start_like = np.zeros((ntemps, nwalkers))
        while np.std(start_like) < 5.0:
            start_params = injection_params_L_frame * (
                1.0
                + factor
                * tmp_cov[None, None, :]
                * np.random.randn(ntemps, nwalkers, ndims["mbh"])
            )
            start_like = np.array(
                [
                    like_fn.get_like(start_params_i)
                    for start_params_i in start_params.reshape(-1, ndims["mbh"])
                ]
            ).reshape(start_params.shape[:-1])
            factor *= 1.5
            print(np.std(start_like))

        start_state = State({"mbh": start_params})
        assert np.all(
            ~np.isinf(priors["mbh"].logpdf(start_params.reshape(-1, ndims["mbh"])))
        )

    het_update = HetUpdate(like_fn)

    periodic = {"mbh": {4: 2 * np.pi, 8: 2 * np.pi, 9: np.pi}}
    sampler = EnsembleSampler(
        nwalkers,
        ndims,
        like_func_wrap,
        priors,
        args=[like_fn],
        moves=moves,
        # nleaves_max={"mbh": 1},
        backend=file_store,
        periodic=periodic,
        branch_names=["mbh"],
        tempering_kwargs={"ntemps": ntemps, "Tmax": np.inf},
        update_fn=het_update,
        update_iterations=8,
    )
    nsteps_saved = 2000
    sampler.run_mcmc(start_state, nsteps_saved, progress=True, thin_by=25)
