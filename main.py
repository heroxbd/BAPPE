import numpy as np
import h5py as h5
import pandas as pd
import math
from zernike import RZern
from scipy.special import legendre
import uproot
import awkward as ak
import wf_func as wff
import pickle
from tqdm import tqdm
import pre
import pmt

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

thetas = np.linspace(0, math.pi, 101)
phis = np.linspace(0, 2 * math.pi, 101)
rs = np.linspace(0, 1, 101)

pmt_poss = pmt.pmt_pos()


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


mixed_rs = np.repeat(rs, len(rs) ** 2)
mixed_thetas = np.repeat(
    np.array([np.repeat(thetas, len(rs))]).T, len(rs), axis=1
).T.flatten()
mixed_phis = np.repeat(np.array([phis]).T, len(rs) ** 2, axis=1).T.flatten()

mixed_xs, mixed_ys, mixed_zs = sph2cart(mixed_rs, mixed_thetas, mixed_phis)


def rthetas(xs, ys, zs, pmt_pos):
    vertex_poss = np.array([xs, ys, zs]).T
    vertex_poss_norm = np.linalg.norm(vertex_poss, axis=1)
    vertex_poss_norm = vertex_poss_norm.reshape(len(vertex_poss_norm), 1)
    vertex_poss = np.where(
        vertex_poss_norm == 0, [0, 0, 0], vertex_poss / vertex_poss_norm
    )
    pmt_pos_norm = np.linalg.norm(pmt_pos)
    pmt_pos /= pmt_pos_norm
    thetas = np.arccos(np.clip(np.einsum("ij, j -> i", vertex_poss, pmt_pos), -1, 1))
    return thetas


rel_thetas = [
    rthetas(mixed_xs, mixed_ys, mixed_zs, pmt_poss[id]) for id in tqdm(range(30))
]

zs2 = np.array(
    [
        [cart.Zk(v, mixed_rs, rel_thetas[id]) for v in range(nr)]
        for id in tqdm(range(30))
    ]
)

for _, trig in ent:
    total_psy = np.ones(101 ** 3)
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
        pys = psy_star / np.prod(
            np.where(smmse != 0, uniform_probe_pre, 1 - uniform_probe_pre), axis=1
        )
        ts2 = (pet / 175) - 1
        lt2 = np.array([legendre(v)(ts2) for v in range(nt)])
        lt2[:, pet > 350] = 0
        # probe_func = np.exp(np.einsum("ij,ik,jl->kl", almn, lt2, zs2[channelid]))
        probe_func = np.exp(np.dot(np.dot(lt2.T, almn), zs2[channelid])) * (
            pet[1] - pet[0]
        )
        print(np.max(probe_func))
        assert not np.any(probe_func > 1)
        psv = np.prod(
            np.einsum("ij,jl->ilj", smmse, probe_func)
            + (1 - np.einsum("ij,jl->ilj", 1 - smmse, probe_func)),
            axis=2,
        )
        total_psy *= np.einsum("i,il->l", pys, psv)
    ind = np.argmax(total_psy)
    print("({}, {}, {})".format(mixed_rs[ind], mixed_thetas[ind], mixed_phis[ind]))
