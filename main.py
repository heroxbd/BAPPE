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
    ipt["SimTriggerInfo/PEList"][("TriggerNo", "PMTId", "PulseTime")][()]
)

dnoise = np.log(1e-5)  # dark noise rate is 1e-5 ns^{-1}
y0 = np.arctan((0 - 0.99) * 1e9)


def radius(t):
    return (np.arctan((t - 0.99) * 1e9) - y0) * 100

def log_prob(value):
    """
    inputs from the global scope:
    1. pys: P(w | s).
    2. smmses: selection indicators.
    3. pets: possible hit times given by lucyddm
    4. lt: legendre values of the whole timing intervals.
    """
    x, y, z = value[:3]
    r = np.sqrt(x * x + y * y + z * z)
    rths = rtheta(x, y, z, PMT)
    res = 0.0

    zs_radial = np.array([cart.radial(v, r) for v in zo])
    almn[0, 0] = a00 + value[3]

    zs_angulars = np.array([cart.angular(v, rths) for v in zo])

    zs = zs_radial.reshape(-1, 1) * zs_angulars

    nonhit = np.sum(np.exp(lt.T @ almn @ zs), axis=0)
    N = np.empty_like(PMT)
    N[pmt_ids] = [x for x in map(lambda x: x.shape[1], smmses)]
    nonhit_PMT = np.setdiff1d(PMT, pmt_ids)
    N[nonhit_PMT] = 0

    res = np.sum(N * np.log(nonhit) - nonhit)
    return res - radius(x * x + y * y + z * z)


nevents = len(PE.groupby("TriggerNo"))

rec = np.empty((nevents, 4))

for ie, trig in PE.groupby("TriggerNo"):
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
        np.array((0, 0, 0, 0), dtype=np.float),
        method="Powell",
        bounds=((-1, 1), (-1, 1), (-1, 1), (None, None)),
    )
    print(x.x)
    rec[ie] = x.x
