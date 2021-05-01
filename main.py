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
from scipy.optimize import minimize
from numba import njit

with open("electron-2.pkl", "rb") as f:
    coef = pickle.load(f)

basename = "electron-5"

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


ts = np.linspace(-1, 1, 351)
lt = np.array([legendre(v)(ts) for v in range(nt)])

PMT = np.arange(30, dtype=np.uint8)

PE = pd.DataFrame.from_records(
    ipt["SimTriggerInfo/PEList"][("TriggerNo", "PMTId", "PulseTime")][:500]
)

dnoise = np.log(1e-5)  # dark noise rate is 1e-5 ns^{-1}
y0 = np.arctan((0 - 0.99) * 1e9)


def radius(t):
    return (np.arctan((t - 0.99) * 1e9) - y0) * 100


# @njit
def legval(x, c):
    """
    stole from the numerical part of numpy.polynomial.legendre

    """
    breakpoint()
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


def log_prob(value):
    """
    inputs from the global scope:
    1. pys: P(w | s).
    2. smmses: selection indicators.
    3. pets: possible hit times given by lucyddm
    4. lt: legendre values of the whole timing intervals.
    """
    t0 = value[0]
    x, y, z = value[1:4]
    r = np.sqrt(x * x + y * y + z * z)
    rths = rtheta(x, y, z, PMT)
    res = 0.0

    zs_radial = np.array([cart.radial(v, r) for v in zo])
    almn[0, 0] = a00 + value[4]

    zs_angulars = np.array([cart.angular(v, rths) for v in zo])

    zs = zs_radial.reshape(-1, 1) * zs_angulars

    nonhit = np.sum(np.exp(lt.T @ almn @ zs), axis=0)
    nonhit_PMT = np.setdiff1d(PMT, pmt_ids)

    for i, hit_PMT in enumerate(pmt_ids):
        probe_func = np.empty_like(pets[i])
        ts2 = (pets[i] - t0) / 175 - 1
        t_in = np.logical_and(ts2 > -1, ts2 < 1)  # inside time window
        if np.any(t_in):
            lt2 = np.polynomial.legendre.legval(ts2[t_in], np.eye(nt))
            probe_func[t_in] = np.logaddexp(lt2.T @ almn @ zs[:, hit_PMT], dnoise)
        probe_func[np.logical_not(t_in)] = dnoise
        psv = np.sum(smmses[i] * (probe_func + np.log(dpets[i])), axis=1)
        psv -= nonhit[hit_PMT]
        lprob = logsumexp(psv + pys[i])
        res += lprob
    res -= np.sum(nonhit[nonhit_PMT])
    print(t0, x, y, z, np.exp(value[4]), res)
    return np.array(res) - radius(x * x + y * y + z * z)


for _, trig in PE.groupby("TriggerNo"):
    smmses = []
    pys = []
    pets = []
    dpets = []
    pmt_ids = trig["PMTId"].unique()
    for _, PMT_hit in trig.groupby("PMTId"):
        smmses.append(np.ones((1, len(PMT_hit)), dtype=int))
        pys.append(1)
        pets.append(PMT_hit["PulseTime"].values)
        dpets.append(1)

    x = minimize(
        lambda z: -log_prob(z),
        np.array((0, 0, 0, 0, 0), dtype=np.float),
        method="Powell",
        bounds=((-5, 5), (-1, 1), (-1, 1), (-1, 1), (None, None)),
    )
    breakpoint()
