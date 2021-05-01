import numpy as np
import h5py as h5
import pandas as pd
import math
from jzernike import RZern
from scipy.special import legendre, logsumexp
import uproot
import awkward as ak
import wf_func as wff
import pickle
from tqdm import tqdm
import pre
import pmt
import numpyro
import jax
import jax.numpy as jnp
import jax.scipy as jscipy

from scipy.optimize import minimize


numpyro.set_platform("cpu")
from numpyro import distributions as dist

with open("electron-2.pkl", "rb") as f:
    coef = pickle.load(f)

basename = "electron-5"

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
ppos_norm = jnp.linalg.norm(pmt_poss, axis=1)
ppos_norm = ppos_norm.reshape((len(ppos_norm), 1))
pmt_poss /= ppos_norm

def sph2cart(r, theta, phi):
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return x, y, z


def cart2sph(x, y, z):
    xy2 = x ** 2 + y ** 2
    r = jnp.sqrt(xy2 + z ** 2)
    theta = jnp.arctan2(z, jnp.sqrt(xy2))
    phi = jnp.arctan2(y, x)
    return r, theta, phi


def rtheta(x, y, z, pmt_ids):
    vpos = jnp.array([x, y, z])
    vpos_norm = jnp.clip(jnp.linalg.norm(vpos), 1e-6)
    vpos /= vpos_norm
    ppos = pmt_poss[pmt_ids]
    theta = jnp.arccos(jnp.clip(jnp.dot(vpos, ppos.T), -1, 1))
    return theta


ts = np.linspace(-1, 1, 351)
lt = np.array([legendre(v)(ts) for v in range(nt)])

PMT = np.arange(30, dtype=np.uint8)

PE = pd.DataFrame.from_records(ipt['SimTriggerInfo/PEList'][("TriggerNo", "PMTId", "PulseTime")][:500])

dnoise = np.log(1e-5) # dark noise rate is 1e-5 ns^{-1}
y0 = np.arctan((0 - 0.99)*1e9)
def radius(t):
    return (np.arctan((t - 0.99)*1e9)-y0)*100

class probe(dist.Distribution):
    support = dist.constraints.unit_interval

    def __init__(self):
        super(probe, self).__init__(batch_shape=(3))

    @numpyro.distributions.util.validate_sample
    def log_prob(self, value):
        """
        inputs from the global scope:
        1. pys: P(w | s).
        2. smmses: selection indicators.
        3. pets: possible hit times given by lucyddm
        4. lt: legendre values of the whole timing intervals.
        """
        t0 = value[0]
        x, y, z = value[1:4]
        rths = rtheta(x, y, z, PMT)
        res = 0.0

        zs_radial = jnp.array([cart.radial(v, r) for v in zo])
        almnE = jax.ops.index_update(almn, jax.ops.index[0, 0], a00 + value[4])

        zs_angulars = jnp.array([cart.angular(v, rths) for v in zo])

        zs = zs_radial.reshape(-1, 1) * zs_angulars

        nonhit = jnp.sum(jnp.exp(lt.T @ almnE @ zs), axis=0)
        nonhit_PMT = np.setdiff1d(PMT, pmt_ids)

        for i, hit_PMT in enumerate(pmt_ids):
            probe_func = np.empty_like(pets[i])
            ts2 = (pets[i] - t0) / 175 - 1
            t_in = np.logical_and(ts2 > -1, ts2 < 1) # inside time window
            if np.any(t_in):
                lt2 = jnp.array([legendre(v)(ts2[t_in]) for v in range(nt)])
                probe_func[t_in] = jnp.logaddexp(lt2.T @ almnE @ zs[:, hit_PMT], dnoise)
            probe_func[np.logical_not(t_in)] = dnoise
            psv = jnp.sum(smmses[i] * (probe_func + jnp.log(dpets[i])), axis=1)
            psv -= nonhit[hit_PMT]
            lprob = jscipy.special.logsumexp(psv + pys[i])
            res += lprob
        res -= np.sum(nonhit[nonhit_PMT])
        print(t0, x, y, z, np.exp(value[4]), res)
        return np.array(res) - radius(x*x+y*y+z*z)


xprobe = probe()

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
        lambda z: -xprobe.log_prob(z),
        np.array((0, 0, 0, 0, 0), dtype=np.float),
        method="Powell",
        bounds=((-5, 5), (-1, 1), (-1, 1), (-1, 1), (None, None)),
    )
