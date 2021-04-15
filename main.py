import numpy as np
import h5py
import uproot
import awkward as ak
import wf_func as wff

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
opdt = np.dtype(
    [
        ("TriggerNo", np.uint32),
        ("ChannelID", np.uint32),
        ("HitPosInWindow", np.uint16),
        ("Charge", np.float64),
    ]
)

fipt = ""
with h5py.File(fipt, "r", libver="latest", swmr=True) as ipt:
    ent = ipt["Readout"]["Waveform"][()]
    ent = ent[ent["ChannelID"] < 30]
    l = len(ent)
    print("{} waveforms will be computed".format(l))
    assert np.all(pedcid == ent["ChannelID"]), "Files do not match!"
    leng = len(ent[0]["Waveform"])
    assert leng >= len(spe_pre[0]["spe"]), "Single PE too long which is {}".format(
        len(spe_pre[0]["spe"])
    )

Thres = 0.1

leng = len(ent[0]["Waveform"])
dt = np.zeros(l * leng, dtype=opdt)
start = 0
end = 0

for i in range(l):
    wave = (ent[i]["Waveform"] - pedestal[i]) * spe_pre[ent[i]["ChannelID"]]["epulse"]
    A, wave, pet, mu, n = wff.initial_params(
        wave, spe_pre[ent[i]["ChannelID"]], Thres, 4, 3
    )
    factor = np.linalg.norm(spe_pre[ent[i]["ChannelID"]]["spe"])
    A = A / factor
    gmu = spe_pre[ent[i]["ChannelID"]]["spe"].sum()
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
        np.repeat(min(-1e-3 + 1, mu / len(pet)), len(pet)),
        spe_pre[ent[i]["ChannelID"]]["std"] ** 2,
        (40.0 * factor / gmu) ** 2,
        factor,
        20,
        stop=0,
    )
    pet, pwe = wff.clip(pet, xmmse_star[0] / factor * gmu, 0)
