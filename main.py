import numpy as np
import pandas as pd
import math
from zernike import RZern
from scipy.special import legendre
import uproot
import awkward as ak
import wf_func as wff
import pickle

with open("electron-2.pkl", "rb") as f:
    coef = pickle.load(f)

baseline_file = ""
with uproot.open(baseline_file) as ped:
    pedestal = ak.to_numpy(
        ak.flatten(ped["SimpleAnalysis"]["ChannelInfo.Pedestal"].array())
    )
    pedcid = ak.to_numpy(
        ak.flatten(ped["SimpleAnalysis"]["ChannelInfo.ChannelId"].array())
    )


spe_file = ""

spe_pre = wff.read_model(spe_file)

fipt = ""
ent = pd.read_hdf(fipt, "Readout/Waveform")
ent = ent[ent["ChannelID"] < 30]
print("{} waveforms will be computed".format(len(ent)))
assert np.all(pedcid == ent["ChannelID"]), "Files do not match!"
leng = len(ent[0]["Waveform"])
assert leng >= len(spe_pre[0]["spe"]), "Single PE too long which is {}".format(
    len(spe_pre[0]["spe"])
)

ent["Pedestal"] = pedestal
ent = ent.groupby(by=["TriggerNo"])

Thres = 0.1

leng = len(ent[0]["Waveform"])

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

thetas2 = np.linspace(0, 2 * math.pi, 101)
rs2 = np.linspace(0, 1, 101)
zs2 = np.array(
    [[[cart.Zk(v, r, theta) for r in rs2] for theta in thetas2] for v in range(nr)]
)

for _, trig in ent:
    total_psy = np.ones((100, 100))
    for pe in trig.iloc:
        wave = (pe["Waveform"] - pe["Pedestal"]) * spe_pre[pe["ChannelID"]]["epulse"]
        A, wave, pet, mu, n = wff.initial_params(
            wave, spe_pre[pe["ChannelID"]], Thres, 4, 3
        )
        factor = np.linalg.norm(spe_pre[pe["ChannelID"]]["spe"])
        A = A / factor
        gmu = spe_pre[pe["ChannelID"]]["spe"].sum()
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
            spe_pre[pe["ChannelID"]]["std"] ** 2,
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
        ts2 = (
            np.concatenate([np.linspace(i, i + 1, num=(n + 1)) for i in pet]) / 175
        ) - 1
        lt2 = np.array([legendre(v)(ts2) for v in range(nt)])
        probe_func = np.exp(np.einsum("ij,ik,jlm->lmk", almn, lt2, zs2))
        psv = np.prod(
            np.einsum("ij,lmj->ilmj", smmse, probe_func)
            + (1 - np.einsum("ij,lmj->ilmj", 1 - smmse, probe_func)),
            axis=3,
        )
        total_psy *= np.einsum("i,ilm->lm", pys, psv)
    ind = np.unravel_index(np.argmax(total_psy), total_psy.shape)
    print("({}, {})".format(rs2[ind[1]], thetas2[ind[0]]))
