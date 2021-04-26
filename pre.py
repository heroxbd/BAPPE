import numpy as np
import pandas as pd
import pmt
from tqdm import tqdm
import math


def __pre_ana(filename):
    print("Reading file...")
    vertices = pd.read_hdf(filename, "SimTruth/SimTruth")
    petruth = pd.read_hdf(filename, "SimTriggerInfo/PEList")
    truth = pd.read_hdf(filename, "SimTriggerInfo/TruthList")

    print("Merging...")
    vertices = pd.merge(
        vertices, truth, on=["RunNo", "SegmentId", "VertexId"], how="outer"
    )
    events = pd.merge(
        vertices, petruth, on=["RunNo", "SegmentId", "TriggerNo"], how="outer"
    )
    events = events[events["SegmentId"] >= 0]
    pets = events.groupby(
        by=["RunNo", "SegmentId", "VertexId", "TriggerNo"], dropna=False
    )

    print("Pre-calculating...")
    npmt = 30

    pmt_poss = pmt.pmt_pos()
    assert len(pmt_poss) == npmt

    return pmt_poss, pets


def __thetas(xs, ys, zs, pmt_ids, pmt_poss):
    vertex_poss = np.array([xs, ys, zs]).T
    vertex_poss_norm = np.linalg.norm(vertex_poss, axis=1)
    vertex_poss_norm = vertex_poss_norm.reshape(len(vertex_poss_norm), 1)
    vertex_poss /= vertex_poss_norm
    pmt_pos_by_ids = pmt_poss[pmt_ids]
    pmt_pos_by_ids_norm = np.linalg.norm(pmt_pos_by_ids, axis=1)
    pmt_pos_by_ids_norm = pmt_pos_by_ids_norm.reshape(len(pmt_pos_by_ids_norm), 1)
    pmt_pos_by_ids /= pmt_pos_by_ids_norm
    thetas = np.arccos(
        np.clip(np.einsum("ij, ij -> i", vertex_poss, pmt_pos_by_ids), -1, 1)
    )
    return thetas


def pre_ana_group(filename, optic=False):
    pmt_poss, pets = __pre_ana(filename)

    pes = pd.DataFrame([vs[["x", "y", "z"]].iloc[0] for _, vs in tqdm(pets)])
    assert len(pes) == len(pets)

    xs = np.array(pes["x"]).repeat(30)
    ys = np.array(pes["y"]).repeat(30)
    zs = np.array(pes["z"]).repeat(30)

    pmt_ids = np.repeat(np.array([np.arange(30)]).T, len(pes), axis=1).T.flatten()
    thetas = __thetas(xs, ys, zs, pmt_ids, pmt_poss)

    rs = np.sqrt(np.array(xs) ** 2 + np.array(ys) ** 2 + np.array(zs) ** 2)
    rs /= 645.0
    n = len(rs)

    tbins = np.linspace(-1, 1, 201)

    tgs = np.array(
        [
            [
                np.histogram(
                    (
                        np.array(
                            (
                                vs[vs["PMTId"] == i]["HitTime"]
                                - vs[vs["PMTId"] == i]["photonTime"]
                            )
                            if optic
                            else vs[vs["PMTId"] == i]["PulseTime"]
                        )
                    )
                    / 175
                    - 1,
                    bins=tbins,
                )[0]
                for i in range(30)
            ]
            for _, vs in tqdm(pets)
        ]
    ).reshape(len(pets) * 30, 200)
    assert len(tgs) == n
    return n, rs, thetas, tbins, tgs, pmt_ids


def pre_ana_concat(filename, optic=False):
    pmt_poss, pets = __pre_ana(filename)
    pets = pd.concat([vs for _, vs in tqdm(pets)])

    xs = np.array(pets["x"])
    ys = np.array(pets["y"])
    zs = np.array(pets["z"])
    ts = (
        np.array((pets["HitTime"] - pets["photonTime"]) if optic else pets["PulseTime"])
    ) / 175 - 1

    breakpoint()

    pmt_ids = np.array(pets["PMTId"], dtype=int)
    thetas = __thetas(xs, ys, zs, pmt_ids, pmt_poss)

    rs = np.sqrt(np.array(xs) ** 2 + np.array(ys) ** 2 + np.array(zs) ** 2)
    rs /= 645.0
    n = len(rs)
    return n, rs, thetas, ts
