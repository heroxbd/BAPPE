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


numpyro.set_platform("gpu")
from numpyro import distributions as dist

with open("electron-2.120.pkl", "rb") as f:
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

almn = np.zeros((nt, nr))
for i in range(nt):
    for j in np.concatenate(([0], range(1, nr, 2))):
        if i == 0 and j == 0:
            almn[i, j] = coef["Intercept"]
        elif i == 0:
            almn[i, j] = coef["Z{}".format(j)]
        elif j == 0:
            almn[i, j] = coef["L{}".format(i)]
        else:
            almn[i, j] = coef["Z{}_L{}".format(j, i)]


pmt_poss = pmt.pmt_pos()


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
    ppos_norm = jnp.linalg.norm(ppos, axis=1)
    ppos_norm = ppos_norm.reshape((len(ppos_norm), 1))
    ppos /= ppos_norm
    theta = jnp.arccos(jnp.clip(jnp.dot(vpos, ppos.T), -1, 1))
    return theta


class probe(dist.Distribution):
    support = dist.constraints.unit_interval

    def __init__(self):
        super(probe, self).__init__()

    @numpyro.distributions.util.validate_sample
    def log_prob(self, value):
        r = value[0]
        theta = value[1] * math.pi
        phi = value[2] * math.pi * 2
        x, y, z = sph2cart(r, theta, phi)
        rths = rtheta(x, y, z, pmt_ids)
        res = 1.0
        zs_radial = jnp.array([cart.radial(v, r) for v in range(nr)])
        zs_angulars = jnp.array([cart.angular(v, rths) for v in range(nr)])
        for i in range(len(pmt_ids)):
            zs = zs_radial * zs_angulars[:, i]
            probe_func = jnp.exp(jnp.dot(jnp.dot(lt2s[i].T, almn), zs) * dpets[i])
            psv = jnp.prod(
                smmses[i] * probe_func + (1 - (1 - smmses[i]) * probe_func),
                axis=1,
            )
            res *= jnp.dot(pys[i], psv)
        return jnp.log(res)


xprobe = probe()


def vertex():
    return numpyro.sample("r", xprobe)


rng_key = jax.random.PRNGKey(8162)

for _, trig in ent:
    pmt_ids = np.array(trig["ChannelID"], dtype=int)
    smmses = []
    pys = []
    lt2s = []
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
            psy_star
            / np.prod(
                np.where(smmse != 0, uniform_probe_pre, 1 - uniform_probe_pre), axis=1
            )
        )
        ts2 = (pet / 175) - 1
        lt2 = np.array([legendre(v)(ts2) for v in range(nt)])
        lt2[:, pet > 350] = 0
        lt2s.append(lt2)
        dpets.append(pet[1] - pet[0])

    nuts_kernel = numpyro.infer.NUTS(
        vertex,
        init_strategy=numpyro.infer.initialization.init_to_value(
            values={"r": jnp.array([0.0, 0.0, 0.0])}
        ),
    )
    mcmc = numpyro.infer.MCMC(
        nuts_kernel, num_samples=200, num_warmup=50, progress_bar=True
    )
    mcmc.run(rng_key, extra_fields=("accept_prob", "potential_energy"))

    sa = mcmc.get_samples()["r"]
    print(sa)
