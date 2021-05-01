import numpy as np
import h5py as h5
import pandas as pd
import math
from zernike import RZern
from scipy.special import legendre, logsumexp
import uproot
import awkward as ak
import wf_func as wff
import pickle
from tqdm import tqdm
import pre
import pmt
from scipy.optimize import minimize, minimize_scalar
from numba import njit

with open("electron-2.pkl", "rb") as f:
    coef = pickle.load(f)

basename = "electron-6"

baseline_file = "{}.baseline.root".format(basename)
with uproot.open(baseline_file) as ped:
    pedestal = ak.to_numpy(
        ak.flatten(ped["SimpleAnalysis"]["ChannelInfo.Pedestal"].array())
    )
    pedcid = ak.to_numpy(
        ak.flatten(ped["SimpleAnalysis"]["ChannelInfo.ChannelId"].array())
    )


spe_file = "{}.spe.h5".format(basename)

spe_pre = wff.read_model(spe_file)

fipt = "{}.h5".format(basename)
ipt = h5.File(fipt, "r")
ent = ipt["Readout/Waveform"]
ent = ent[ent["ChannelID"] < 30]
print("{} waveforms will be computed".format(len(ent)))
assert np.all(pedcid == ent["ChannelID"]), "Files do not match!"
leng = len(ent[0]["Waveform"])
assert leng >= len(spe_pre[0]["spe"]), "Single PE too long which is {}".format(
    len(spe_pre[0]["spe"])
)

waveforms = ent["Waveform"]
ent = pd.DataFrame(
    data={
        "id": range(0, len(ent)),
        "TriggerNo": ent["TriggerNo"],
        "ChannelID": ent["ChannelID"],
    }
)
ent["Pedestal"] = pedestal
ent = ent.groupby(by=["TriggerNo"])

Thres = 0.1

fipt = "{}.h5".format(basename)
ipt = h5.File(fipt, "r")

nt = 80
nr = 120
cart = RZern(20)

almn = np.zeros((nt, nr // 2 + 1))

zo = np.concatenate(([0], range(1, nr, 2)))  # zernike orders
for i in range(nt):
    for j in zo:
        if i == 0 and j == 0:
            a00 = coef["Intercept"]
        elif j == 0:
            almn[i, j] = coef["L{}".format(i)]
        elif i == 0:
            almn[i, (j + 1) // 2] = coef["Z{}".format(j)]
        else:
            almn[i, (j + 1) // 2] = coef["Z{}_L{}".format(j, i)]
zrho = cart.rhotab[zo, :]

pmt_poss = pmt.pmt_pos()
ppos_norm = np.linalg.norm(pmt_poss, axis=1)
ppos_norm = ppos_norm.reshape((len(ppos_norm), 1))
pmt_poss /= ppos_norm


def sph2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cart2sph(x, y, z):
    xy2 = x ** 2 + y ** 2
    r = np.sqrt(xy2 + z ** 2)
    theta = np.arctan2(z, np.sqrt(xy2))
    phi = np.arctan2(y, x)
    return r, theta, phi


def rtheta(x, y, z, pmt_ids):
    vpos = np.array([x, y, z])
    vpos_norm = np.clip(np.linalg.norm(vpos), 1e-6, None)
    vpos /= vpos_norm
    ppos = pmt_poss[pmt_ids]
    theta = np.arccos(np.clip(np.dot(vpos, ppos.T), -1, 1))
    return theta


PMT = np.arange(30, dtype=np.uint8)

PE = pd.DataFrame.from_records(
    ipt["SimTriggerInfo/PEList"][("TriggerNo", "PMTId", "PulseTime")][()]
)

dnoise = np.log(1e-5)  # dark noise rate is 1e-5 ns^{-1}
y0 = np.arctan((0 - 0.99) * 1e9)


def radius(t):
    return (np.arctan((t - 0.99) * 1e9) - y0) * 100


@njit
def polyval(p, x):
    y = np.zeros(p.shape[1])
    for i in range(len(p)):
        y = y * x + p[i]
    return y


@njit
def radial(coefnorm, rhotab, k, rho):
    return coefnorm[k] * polyval(rhotab[k, :].T, rho)


@njit
def angular(m, theta):
    return np.cos(m * theta)


@njit
def legval(x, c):
    """
    stole from the numerical part of numpy.polynomial.legendre

    """
    if len(c) == 1:
        return c[0]
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * x * (2 * nd - 1)) / nd
    return c0 + c1 * x

ts = np.linspace(-1, 1, 351)
lt = np.polynomial.legendre.legval(ts, np.eye(nt))

leg_order = np.eye(nt).reshape(nt, nt, 1)

def log_prob(x, y, z, t0, logE):
    """
    inputs from the global scope:
    1. pys: P(w | s).
    2. smmses: selection indicators.
    3. pets: possible hit times given by lucyddm
    4. lt: legendre values of the whole timing intervals.
    """
    r = np.sqrt(x * x + y * y + z * z)
    rths = rtheta(x, y, z, PMT)
    res = 0.0

    zs_radial = radial(cart.coefnorm, cart.rhotab, zo, r)
    almn[0, 0] = a00 + logE
    zs_angulars = angular(cart.mtab[zo], rths.reshape(-1, 1))

    zs = zs_radial * zs_angulars

    nonhit = np.sum(np.exp(lt.T @ almn @ zs.T), axis=0)
    nonhit_PMT = np.setdiff1d(PMT, pmt_ids)

    for i, hit_PMT in enumerate(pmt_ids):
        probe_func = np.empty_like(pets[i])
        ts2 = (pets[i] - t0) / 175 - 1
        t_in = np.logical_and(ts2 > -1, ts2 < 1)  # inside time window
        if np.any(t_in):
            lt2 = legval(ts2[t_in], leg_order)
            probe_func[t_in] = np.logaddexp(lt2.T @ almn @ zs[hit_PMT], dnoise)
        probe_func[np.logical_not(t_in)] = dnoise
        psv = np.sum(smmses[i] * (probe_func + np.log(dpets[i])), axis=1)
        psv -= nonhit[hit_PMT]
        lprob = logsumexp(psv + pys[i])
        res += lprob
    res -= np.sum(nonhit[nonhit_PMT])
    return res - radius(r)


nevents = len(PE.groupby("TriggerNo"))

rec = np.empty((3002, 5))

nevt = 0
for ie, trig in ent:
    pmt_ids = np.array(trig["ChannelID"], dtype=int)
    smmses = []
    pys = []
    pets = []
    dpets = []
    for pe in trig.iloc:
        channelid = int(pe["ChannelID"])
        wave = (waveforms[int(pe["id"])] - pe["Pedestal"]) * spe_pre[channelid][
            "epulse"
        ]
        A, wave, pet, mu, n = wff.initial_params(wave, spe_pre[channelid], Thres, 4, 3)
        factor = np.linalg.norm(spe_pre[channelid]["spe"])
        A = A / factor
        gmu = spe_pre[channelid]["spe"].sum()
        uniform_probe_pre = min(-1e-3 + 1, mu / len(pet))
        probe_pre = np.repeat(uniform_probe_pre, len(pet))
        (
            xmmse,
            xmmse_star,
            psy_star,
            nu_star,
            T_star,
            d_tot_i,
            d_max,
        ) = wff.fbmpr_fxn_reduced(
            wave,
            A,
            probe_pre,
            spe_pre[channelid]["std"] ** 2,
            # TODO: 40.0: 单光电子响应的电荷分布展宽
            (40.0 * factor / gmu) ** 2,
            factor,
            20,
            stop=0,
        )
        smmse = np.where(xmmse_star != 0, 1, 0)
        smmses.append(smmse)
        pys.append(
            nu_star
            - np.sum(
                np.log(np.where(smmse != 0, uniform_probe_pre, 1 - uniform_probe_pre)),
                axis=1,
            )
        )
        pets.append(pet)
        dpets.append(pet[1] - pet[0])

    tx = minimize_scalar(
        lambda t0: -log_prob(0., 0., 0., t0, 0.),
        bracket = (0, 300, 1000)
    )
    
    x = minimize(
        lambda z: -log_prob(*z),
        np.array((0, 0, 0, tx.x, 0), dtype=np.float),
        method="SLSQP",
        bounds=((-1, 1), (-1, 1), (-1, 1), (-5, 1029 - 350), (None, None)),
    )
    rec[ie] = x.x

    print(x.x)
    nevt += 1
    if nevt > 3002:
        break
