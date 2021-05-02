import numpy as np
import h5py as h5
import pandas as pd
from zernike import RZern
import uproot
import awkward as ak
import wf_func as wff
import pickle
import pmt
from scipy.optimize import minimize
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

amn = np.zeros((nt, nr // 2 + 1))

zo = np.concatenate(([0], range(1, nr, 2)))  # zernike orders
for i in range(nt):
    for j in zo:
        if i == 0 and j == 0:
            a00 = coef["Intercept"]
        elif j == 0:
            amn[i, j] = coef["L{}".format(i)]
        elif i == 0:
            amn[i, (j + 1) // 2] = coef["Z{}".format(j)]
        else:
            amn[i, (j + 1) // 2] = coef["Z{}_L{}".format(j, i)]
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
def logsumexp(values, index):
    """Stole from scipy.special.logsumexp

    Parameters
    ----------
    values : array_like Input array.

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.
    """
    a_max = np.max(values)
    s = np.sum(np.exp(values - a_max))
    return np.log(s) + a_max


@njit
def legval(x, n):
    res = np.zeros((n,) + x.shape)
    res[0] = 1
    res[1] = x
    for i in range(2, n):
        res[i] = ((2 * i - 1) * x * res[i - 1] - (i - 1) * res[i - 2]) / i
    return res


ts = np.linspace(-1, 1, 351)
lt = legval(ts, nt)


def log_prob(x, y, z, t0, logE, a_pet, a_pys):
    """
    a_pet: hit times given by LucyDDM and FBMP
    a_pys: log P(w | s).
    - Field "dPEt": log of PE time intervals in 1ns

    inputs from the global scope:
    lt: legendre values of the whole timing intervals.
    """
    r = np.sqrt(x * x + y * y + z * z)
    rths = rtheta(x, y, z, PMT)

    zs_radial = radial(cart.coefnorm, cart.rhotab, zo, r)
    amn[0, 0] = a00 + logE
    zs_angulars = angular(cart.mtab[zo], rths.reshape(-1, 1))

    zs = zs_radial * zs_angulars
    aZ = amn @ zs.T

    nonhit = np.sum(np.exp(lt.T @ aZ))

    ts2 = (a_pet["PEt"] - t0) / 175 - 1
    t_in = np.logical_and(ts2 > -1, ts2 < 1)  # inside time window
    if np.any(t_in):
        tsu, ts_idx = np.unique(ts2[t_in], return_inverse=True)
        lt2 = legval(tsu, nt)
        # 每个 PE 都要使用一次 aZ
        a_pet["probe_func"][t_in] = np.logaddexp(
            np.einsum("ij,ij->j", aZ[:, a_pet["PMTId"][t_in]], lt2[:, ts_idx]), dnoise
        )
    a_pet["probe_func"][np.logical_not(t_in)] = dnoise
    a_pet["probe_func"] += a_pet["dPEt"]  # 每个 PE 都要乘一个区间长度

    lprob = (
        pd.DataFrame.from_records(a_pet)
        .groupby(["PMTId", "PE_config"])["probe_func"]
        .sum()
    )
    # 每个 PE_config 都要乘一个波形分析的 P(w | s) 概率
    lprob += a_pys["pys"]

    # 以 PMTId level=0 算 logsumexp，求和
    hit = lprob.groupby(level=0).agg(logsumexp, engine="numba").sum()
    return hit - nonhit - radius(r)


nevents = len(PE.groupby("TriggerNo"))

rec = np.empty((3002, 5))

nevt = 0
for ie, trig in ent:
    pmt_ids = np.array(trig["ChannelID"], dtype=int)
    pys = []
    pets = []
    for pe in trig.iloc:
        channelid = int(pe["ChannelID"])
        wave = (waveforms[int(pe["id"])] - pe["Pedestal"]) * spe_pre[channelid][
            "epulse"
        ]
        A, wave, pet, mu, n = wff.initial_params(wave, spe_pre[channelid], Thres, 4, 3)
        factor = np.linalg.norm(spe_pre[channelid]["spe"])
        A = A / factor
        gmu = spe_pre[channelid]["spe"].sum()
        uniform_probe_pre = np.clip(mu / len(pet), 1e-3, 1 - 1e-3)
        probe_pre = np.repeat(uniform_probe_pre, len(pet))
        (T_star, nu_star) = wff.fbmpr_fxn_reduced(
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

        config_nPE = np.array(list(map(len, T_star)))
        N_config = np.arange(len(nu_star))
        pet_array = np.empty(
            sum(config_nPE),
            dtype=[
                ("PEt", "f8"),
                ("PMTId", "u4"),
                ("PE_config", "u4"),
                ("probe_func", "f8"),
                ("dPEt", "f8"),
            ],
        )
        pet_array["PMTId"] = channelid
        pet_array["PEt"] = pet[np.concatenate(T_star)]
        pet_array["PE_config"] = np.repeat(N_config, config_nPE)
        # duplicate dPEt to avoid extra merges.
        pet_array["dPEt"] = np.log(pet[1] - pet[0])
        pets.append(pet_array)

        pys_array = np.empty_like(
            nu_star, dtype=[("pys", "f8"), ("PMTId", "u4"), ("PE_config", "u4")]
        )
        pys_array["pys"] = (
            nu_star
            - np.log(uniform_probe_pre) * config_nPE
            - np.log(1 - uniform_probe_pre) * (len(pets) - config_nPE)
        )
        pys_array["PMTId"] = channelid
        pys_array["PE_config"] = N_config
        pys.append(pys_array)

    a_pet = np.concatenate(pets)
    a_pys = np.concatenate(pys)

    tx = np.median(a_pet["PEt"]) - 30
    x = minimize(
        lambda z: -log_prob(*z, a_pet, a_pys),
        np.array((0, 0, 0, tx, 0), dtype=np.float),
        method="SLSQP",
        bounds=((-1, 1), (-1, 1), (-1, 1), (-5, 1029 - 350), (None, None)),
    )
    rec[ie] = x.x

    print(x.x)
    nevt += 1
    if nevt > 200:
        break
