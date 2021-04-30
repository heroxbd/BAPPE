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


# numpyro.set_platform("gpu")
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


ts = np.arange(1029)


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
        r = value[0]
        theta = value[1] * math.pi
        phi = value[2] * math.pi * 2
        t0 = value[3]
        x, y, z = sph2cart(r, theta, phi)
        rths = rtheta(x, y, z, PMT)
        res = 0.0

        zs_radial = jnp.array([cart.radial(v, r) for v in zo])
        almn[0, 0] = a00 + value[4]

        zs_angulars = jnp.array([cart.angular(v, rths) for v in zo])

        zs = zs_radial.reshape(-1, 1) * zs_angulars

        ts1 = (ts - t0) / 175 - 1
        ts1 = ts1[np.logical_and(ts1 >= -1, ts1 <= 1)]
        lt = jnp.array([legendre(v)(ts1) for v in range(nt)])
        nonhit = jnp.sum(jnp.exp(lt.T @ almn @ zs), axis=0)
        nonhit_PMT = np.setdiff1d(PMT, pmt_ids)

        for i, hit_PMT in enumerate(pmt_ids):
            ts2 = (pets[i] - t0) / 175 - 1
            lt2 = jnp.array([legendre(v)(ts2) for v in range(nt)])
            lt2 = jax.ops.index_update(
                lt2, jax.ops.index[:, jnp.logical_or(ts2 < -1, ts2 > 1)], 0
            )
            probe_func = lt2.T @ almn @ zs[:, hit_PMT]
            psv = jnp.sum(smmses[i] * (probe_func + jnp.log(dpets[i])), axis=1)
            psv -= nonhit[hit_PMT]
            lprob = jscipy.special.logsumexp(psv + pys[i])
            res += lprob
        res -= np.sum(nonhit[nonhit_PMT])
        return np.array(res)


xprobe = probe()


def vertex():
    return numpyro.sample("r", xprobe)


rng_key = jax.random.PRNGKey(8162)

PMT = np.arange(30, dtype=np.uint8)
for _, trig in ent:
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

    x = minimize(
        lambda z: -xprobe.log_prob(z),
        (0, 0, 0, 0, 0),
        method="Powell",
        bounds=((0, 1), (0, 1), (0, 1), (0, 1029 - 350), (None, None)),
    )
    print(x)
