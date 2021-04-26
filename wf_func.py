# -*- coding: utf-8 -*-

import os
import math

import numpy as np
from scipy.signal import convolve
from scipy.interpolate import interp1d
import h5py


def lucyddm(waveform, spe_pre, iterations=100):
    """Lucy deconvolution
    Parameters
    ----------
    waveform : 1d array
    spe : 1d array
        point spread function; single photon electron response
    iterations : int

    Returns
    -------
    signal : 1d array

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    .. [2] https://github.com/scikit-image/scikit-image/blob/master/skimage/restoration/deconvolution.py#L329
    """
    spe = np.append(np.zeros(len(spe_pre) - 2 * 9 - 1), np.abs(spe_pre))
    waveform = np.where(waveform < 0, 0.0001, waveform)
    waveform = waveform / np.sum(spe)
    wave_deconv = waveform.copy()
    spe_mirror = spe[::-1]
    for _ in range(iterations):
        relative_blur = waveform / np.convolve(wave_deconv, spe, mode="same")
        wave_deconv *= np.convolve(relative_blur, spe_mirror, mode="same")
    return np.arange(0, len(waveform) - 9), wave_deconv[9:]


def read_model(spe_path):
    with h5py.File(spe_path, "r", libver="latest", swmr=True) as speFile:
        cid = speFile["SinglePE"].attrs["ChannelID"]
        epulse = speFile["SinglePE"].attrs["Epulse"]
        spe = speFile["SinglePE"].attrs["SpePositive"]
        std = speFile["SinglePE"].attrs["Std"]
        spe_pre = {}
        for i in range(len(spe)):
            peak_c = np.argmax(spe[i])
            t = np.argwhere(spe[i][peak_c:] < 0.1).flatten()[0] + peak_c
            mar_l = np.sum(spe[i][:peak_c] < 5 * std[i])
            mar_r = np.sum(spe[i][peak_c:t] < 5 * std[i])
            spe_pre_i = {
                "spe": spe[i],
                "epulse": epulse,
                "peak_c": peak_c,
                "mar_l": mar_l,
                "mar_r": mar_r,
                "std": std[i],
            }
            spe_pre.update({cid[i]: spe_pre_i})
    return spe_pre


def clip(pet, pwe, thres):
    if len(pet[pwe > thres]) == 0:
        pet = np.array([pet[np.argmax(pwe)]])
        pwe = np.array([1])
    else:
        pet = pet[pwe > thres]
        pwe = pwe[pwe > thres]
    return pet, pwe


def initial_params(wave, spe_pre, Thres, nsp, nstd):
    hitt, char = lucyddm(wave, spe_pre["spe"])
    hitt, char = clip(hitt, char, Thres)
    char = char / char.sum() * np.clip(np.abs(wave.sum()), 1e-6, np.inf)
    tlist = np.unique(
        np.floor(
            np.clip(
                np.hstack(hitt[:, None] + np.arange(-nsp, nsp + 1)), 0, len(wave) - 1
            )
        )
    )

    index_prom = np.hstack([np.argwhere(wave > nstd * spe_pre["std"]).flatten(), hitt])
    left_wave = np.clip(
        index_prom.min() - round(3 * spe_pre["mar_l"]), 0, len(wave) - 1
    )
    right_wave = np.clip(
        index_prom.max() + round(3 * spe_pre["mar_r"]), 0, len(wave) - 1
    )
    wave = wave[left_wave:right_wave]
    mu = wave.sum() / spe_pre["spe"].sum()
    n = max(1, math.ceil(mu))
    ft = interp1d(
        np.arange(0, len(spe_pre["spe"])),
        spe_pre["spe"],
        bounds_error=False,
        fill_value=0,
    )

    tlist = np.sort(np.hstack(tlist[:, None] + np.arange(0, 1, 1 / n)))
    t_auto = np.arange(left_wave, right_wave)[:, None] - tlist
    A = ft((t_auto + np.abs(t_auto)) / 2)

    return A, wave, tlist, mu, n


def fbmpr_fxn_reduced(y, A, p1, sig2w, sig2s, mus, D, stop=0):
    M, N = A.shape

    p = p1.mean()
    nu_true_mean = (
        -M / 2
        - M / 2 * np.log(sig2w)
        - p * N / 2 * np.log(sig2s / sig2w + 1)
        - M / 2 * np.log(2 * np.pi)
        + N * np.log(1 - p)
        + p * N * np.log(p / (1 - p))
    )
    nu_true_stdv = np.sqrt(
        M / 2
        + N * p * (1 - p) * (np.log(p / (1 - p)) - np.log(sig2s / sig2w + 1) / 2) ** 2
    )
    nu_stop = nu_true_mean - stop * nu_true_stdv

    psy_thresh = 1e-4
    P = min(M, 1 + math.ceil(N * p + 1.82138636 * math.sqrt(2 * N * p * (1 - p))))

    T = np.full((P, D), 0)
    nu = np.full((P, D), -np.inf)
    xmmse = np.zeros((P, D, N))
    d_tot = D - 1

    nu_root = (
        -np.linalg.norm(y) ** 2 / 2 / sig2w
        - M * np.log(2 * np.pi) / 2
        - M * np.log(sig2w) / 2
        + np.log(1 - p1).sum()
    )
    Bxt_root = A / sig2w
    betaxt_root = np.abs(sig2s / (1 + sig2s * np.sum(A * Bxt_root, axis=0)))
    nuxt_root = (
        nu_root
        + np.log(betaxt_root / sig2s) / 2
        + 0.5 * betaxt_root * np.abs(np.dot(y, Bxt_root) + mus / sig2s) ** 2
        - 0.5 * mus ** 2 / sig2s
        + np.log(p1 / (1 - p1))
    )

    for d in range(D):
        nuxt = nuxt_root.copy()
        z = y.copy()
        Bxt = Bxt_root.copy()
        betaxt = betaxt_root.copy()
        for p in range(P):
            nustar = max(nuxt)
            nstar = np.argmax(nuxt)
            while np.sum(np.abs(nustar - nu[p, :d]) < 1e-8):
                nuxt[nstar] = -np.inf
                nustar = max(nuxt)
                nstar = np.argmax(nuxt)
            nu[p, d] = nustar
            T[p, d] = nstar
            z = z - A[:, nstar] * mus
            Bxt = Bxt - np.dot(
                betaxt[nstar] * Bxt[:, nstar].copy().reshape(M, 1),
                np.dot(Bxt[:, nstar], A).copy().reshape(1, N),
            )
            assist = np.zeros(N)
            assist[T[: p + 1, d]] = mus + sig2s * np.dot(z, Bxt[:, T[: p + 1, d]])
            xmmse[p, d] = assist
            betaxt = np.abs(sig2s / (1 + sig2s * np.sum(A * Bxt, axis=0)))
            nuxt = (
                nustar
                + np.log(betaxt / sig2s) / 2
                + 0.5 * betaxt * np.abs(np.dot(z, Bxt) + mus / sig2s) ** 2
                - 0.5 * mus ** 2 / sig2s
                + np.log(p1 / (1 - p1))
            )
            nuxt[T[: p + 1, d]] = np.full(p + 1, -np.inf)

        if max(nu[:, d]) > nu_stop:
            d_tot = d + 1
            break
    nu = nu[:, : d + 1].T.flatten()

    # dum = np.sort(nu)[::-1]
    indx = np.argsort(nu)[::-1]
    d_max = math.floor(indx[0] // P) + 1
    nu_max = nu[indx[0]]
    num = int(np.sum(nu > nu_max + np.log(psy_thresh)))
    nu_star = nu[indx[:num]]
    psy_star = np.exp(nu_star - nu_max) / np.sum(np.exp(nu_star - nu_max))
    T_star = [T[: (indx[k] % P) + 1, indx[k] // P] for k in range(num)]
    xmmse_star = np.empty((num, N))
    p1_up = 0
    for k in range(num):
        xmmse_star[k] = xmmse[indx[k] % P, indx[k] // P]
        p1_up = p1_up + psy_star[k] / N * len(T_star[k])

    xmmse = np.dot(psy_star, xmmse_star)

    return xmmse, xmmse_star, psy_star, nu_star, T_star, d_tot, d_max
