# SPDX-License-Identifier: Apache-2.0
"""Wide layer launch (``glm53_flash.gfx950.wide.layer``) on ``tp`` gfx950 GPUs against
torch / the standalone stages:

    core    the KDA task (``kda.standalone``) on the launch's own proj (bitwise)
    o       sum over ranks of core_r @ W_o,r^T (bf16 tolerance); every rank the same bits
    R_mid   ``_mhc_post_torch`` of the launch's o (bitwise)
    x_f, post_f, comb_f   ``_mhc_pre_torch`` + RMSNorm of R_mid
    ids, w  top-8 of sigmoid(x_f @ W_r^T) + bias (torch, the launch's x_f): the picked
            set (near-ties excused), weights; every rank the same bits

    torchrun --nproc-per-node 4 layer_bench.py [--reqs 3,6,16] [--stop fgate] [--time]
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

os.environ.setdefault("SGLANG_USE_AITER", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kimi_k31"))

from mhc_bench import mhc_post_ref, mhc_pre_ref, rmsnorm_ref  # noqa: E402
from tile_bench import bench  # noqa: E402

from sgl_amd_mono.common.peer import PeerBuffer  # noqa: E402
from sgl_amd_mono.glm53_flash.gfx950.config import HIDDEN, MOE_SLOTS, N_EXPERTS, TOP_K  # noqa: E402
from sgl_amd_mono.glm53_flash.gfx950.wide import coll, kda, layer, mhc  # noqa: E402

HC = mhc.HC
LB = -5.0
ONORM_EPS = 1e-5
NH, KD, P, NPROJ = kda.NH, kda.KD, kda.P, kda.NPROJ


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def gather(t):
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    return out


def run(a, reqs, sbf, rank, tp, dev, cpu):
    tt = a.tt
    m = reqs * tt
    slots = reqs + 2
    L = tt
    # replicated: same seed on every rank
    gen = torch.Generator(device=dev).manual_seed(1000 + reqs)

    def rnd(*s, scale=1.0, dt=torch.float32, g=gen):
        return (torch.randn(*s, device=dev, generator=g) * scale).to(dt)

    res_in = rnd(m, HC, HIDDEN, dt=torch.bfloat16)
    mhc_w = []
    for _ in range(2):
        mhc_w.append(dict(
            fn=rnd(24, HC * HIDDEN, scale=0.02),
            scale=torch.tensor([0.8, 0.6, 1.1], device=dev),
            base=rnd(24, scale=0.3),
            norm=(1 + rnd(HIDDEN, scale=0.1)).to(torch.bfloat16),
        ))
    w_r = rnd(N_EXPERTS, HIDDEN, scale=0.02, dt=torch.bfloat16)
    bias_r = rnd(N_EXPERTS, scale=0.5, dt=torch.bfloat16)
    # this rank's heads
    lg = torch.Generator(device=dev).manual_seed(100 + rank)
    w_in = rnd(NPROJ, HIDDEN, scale=0.02, dt=torch.bfloat16, g=lg)
    w_o = rnd(HIDDEN, P, scale=0.02, dt=torch.bfloat16, g=lg)
    conv_w = rnd(3 * P, 4, scale=0.3, g=lg)
    w_fgb = rnd(2, P, KD, scale=0.08, dt=torch.bfloat16, g=lg)
    a_log = torch.log(torch.rand(NH, device=dev, generator=lg) * 8 + 1)
    dt_bias = rnd(P, scale=0.5, g=lg)
    on_w = 1 + rnd(KD, scale=0.1, g=lg)
    conv_st = rnd(slots, 3, 3 * P, scale=0.5, dt=torch.bfloat16, g=lg).transpose(1, 2)
    sdt = torch.bfloat16 if sbf else torch.float32
    state = rnd(slots, NH, KD, KD, scale=0.05, dt=sdt, g=lg)
    idx = torch.arange(1, reqs + 1, device=dev, dtype=torch.int32)
    cs_idx = idx.clone()
    if reqs > 2:
        cs_idx[1] = -1

    def kbufs():
        return dict(
            rawv=torch.zeros(slots, NH, L, KD, device=dev, dtype=torch.bfloat16),
            rawk=torch.zeros(slots, NH, L, KD, device=dev, dtype=torch.bfloat16),
            ring_g=torch.zeros(slots, NH, L, KD, device=dev),
            ring_beta=torch.zeros(slots, NH, L, device=dev),
            iw=torch.zeros(slots, tt, 3, 3 * P, device=dev, dtype=torch.bfloat16).transpose(2, 3),
            core=torch.zeros(m, P, device=dev, dtype=torch.bfloat16),
        )

    mine, ref = kbufs(), kbufs()
    z = lambda *s, dt=torch.float32: torch.zeros(*s, device=dev, dtype=dt)  # noqa: E731
    x_a, post_a, comb_a = z(m, HIDDEN, dt=torch.bfloat16), z(m, HC), z(m, HC * HC)
    proj = z(m, NPROJ, dt=torch.bfloat16)
    acc_o = z(m, HIDDEN, dt=torch.bfloat16)
    res_mid = z(m, HC, HIDDEN, dt=torch.bfloat16)
    x_f, post_f, comb_f = z(m, HIDDEN, dt=torch.bfloat16), z(m, HC), z(m, HC * HC)
    ids, wts, logits = z(m, MOE_SLOTS, dt=torch.int32), z(m, MOE_SLOTS), z(m, N_EXPERTS)
    launch = layer.build(
        m, tt, tp, LB, ONORM_EPS, bool(sbf), True, a.stop, trace=a.trace,
        oneshot=m <= a.oneshot_rows,
    )
    plan = launch.plan
    epoch = z(plan["epoch_words"], dt=torch.int32)
    scratch = z(plan["scratch_words"], dt=torch.int32)
    peers = PeerBuffer(plan["peer_bytes"], cpu, rank, tp, dev)
    peers.bytes.zero_()
    iw_s = mine["iw"].stride()
    ints = (conv_st.stride(0), conv_st.stride(2), iw_s[0], iw_s[1], iw_s[3],
            mine["rawv"].stride(0), mine["rawk"].stride(0), mine["ring_g"].stride(0),
            mine["ring_beta"].stride(0), L)
    fa, ff = mhc_w
    ptrs = (res_in, fa["fn"], fa["scale"], fa["base"], fa["norm"], x_a, post_a, comb_a,
            w_in, proj, conv_w, conv_st, cs_idx, mine["iw"], idx, w_fgb, a_log, dt_bias,
            state, idx, mine["rawv"], mine["rawk"], mine["ring_g"], mine["ring_beta"], idx,
            on_w, mine["core"], w_o, acc_o, ff["fn"], ff["scale"], ff["base"], ff["norm"],
            res_mid, x_f, post_f, comb_f, w_r, bias_r, ids, wts, logits, epoch, scratch)
    pl = [t.data_ptr() for t in ptrs] + [peers.addresses.data_ptr(), rank]

    def go():
        launch(*pl, *ints, stream=torch.cuda.current_stream())

    dist.barrier(cpu)
    go()
    torch.cuda.synchronize()
    errs = {}
    # core: the KDA task alone on our proj
    kl = kda.standalone(reqs, tt, LB, ONORM_EPS, bool(sbf), True)
    kp = (proj, conv_w, conv_st, cs_idx, ref["iw"], idx, w_fgb, a_log, dt_bias, state, idx,
          ref["rawv"], ref["rawk"], ref["ring_g"], ref["ring_beta"], idx, on_w, ref["core"])
    kl(*[t.data_ptr() for t in kp], *ints, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    errs["core"] = rel(mine["core"], ref["core"])
    errs["proj"] = rel(proj, (x_a.float() @ w_in.float().T).to(torch.bfloat16))
    if a.stop != "kda":
        offs, half = coll.layout(m, plan["rb_o"], plan["oproj"].bn, tp, plan["oneshot"])
        par = (epoch[0].item() - 1) % 2
        if plan["oneshot"]:
            os_ = peers.bytes[par * half + offs["os"] :][: tp * m * HIDDEN * 2]
            os_ = os_.view(torch.bfloat16).view(tp, m, HIDDEN).float()
            o = os_[0]
            for p_ in range(1, tp):
                o = o + os_[p_]
            o = o.to(torch.bfloat16)
        else:
            o = peers.bytes[par * half + offs["ag"] :][: m * HIDDEN * 2]
            o = o.view(torch.bfloat16).view(m, HIDDEN).clone()
        cores, wos = gather(mine["core"]), gather(w_o)
        o_ref = sum(c.float() @ w.float().T for c, w in zip(cores, wos)).to(torch.bfloat16)
        errs["o"] = rel(o, o_ref)
        errs["o_ranks"] = max(float((t != o).any()) for t in gather(o))
        if layer.STOPS.index(a.stop) >= 2:
            r_ref = mhc_post_ref(o, res_in, post_a, comb_a)
            errs["R_mid"] = rel(res_mid, r_ref)
            p_ref, c_ref, li_ref = mhc_pre_ref(res_mid, ff["fn"], ff["scale"], ff["base"])
            xr = rmsnorm_ref(li_ref, ff["norm"])
            errs["x_f"] = rel(x_f, xr)
            if errs["x_f"] > 1e-3 and rank == 0:
                print("x_f", x_f.float().norm().item(), "ref", xr.float().norm().item(),
                      "x_a", x_a.float().norm().item(), "rows", x_f.float().norm(dim=1)[:4].tolist(),
                      "x_a vs ref", rel(x_a, xr), flush=True)
            errs["post_f"] = rel(post_f, p_ref)
            errs["comb_f"] = rel(comb_f, c_ref)
            errs["xf_ranks"] = max(float((t != x_f).any()) for t in gather(x_f))
        if a.stop == "route":
            errs.update(route_check(x_f, w_r, bias_r, ids, wts))
            errs["logits"] = rel(logits, x_f.float() @ w_r.float().T)
            errs["route_ranks"] = max(
                float((t != ids).any()) for t in gather(ids)
            ) + max(float((t != wts).any()) for t in gather(wts))
    # repeated launches: the epoch hand-offs, across ranks
    for _ in range(a.launches):
        go()
    torch.cuda.synchronize()
    errs["ep"] = epoch[0].item()
    line = f"rank {rank} reqs {reqs:2d} rows {m:3d} os {int(plan['oneshot'])} state {'bf16' if sbf else 'f32 '} " + "  ".join(
        f"{k} {v:.1e}" if isinstance(v, float) else f"{k} {v}" for k, v in errs.items()
    )
    if a.time:
        dist.barrier(cpu)
        line += f"  layer {bench(lambda _w: go(), [None] * 4):6.1f} us"
    print(line, flush=True)
    if a.trace:
        from sgl_amd_mono.common.plan import BLOCKS
        from sgl_amd_mono.glm53_flash.gfx950.wide.m1a import TRACE_POINTS

        sc = layer.scratch_layout(m, tt, True)
        acc = None
        for _ in range(20):
            go()
            torch.cuda.synchronize()
            tr = scratch[sc["trace"]:][: BLOCKS * TRACE_POINTS * 2].view(torch.int64)
            tr = tr.view(BLOCKS, TRACE_POINTS).double()
            tr = (tr - tr[:, 0].min()) / 100.0
            acc = tr if acc is None else acc + tr
        tr = acc / 20
        names = ["start", "aseam", "agate", "inproj", "kda", "oproj", "reduce", "fseam",
                 "fgate", "epoch", "router", "route", "rwait"]
        for p_, nm in sorted(enumerate(names), key=lambda x: {9: 99, 12: 10.5}.get(x[0], x[0])):
            col = tr[:, p_]
            col = col[(col > -1e5) & (col < 1e5)]
            if col.numel() and rank == 0:
                print(f"    {nm:>7}  median {col.median():6.2f}  max {col.max():6.2f} us", flush=True)
    peers.close()
    tol = {"core": 1e-6, "proj": 1e-2, "o": 1e-2, "o_ranks": 0.0, "R_mid": 1e-6, "x_f": 1e-3,
           "post_f": 1e-5, "comb_f": 1e-5, "xf_ranks": 0.0, "ids_bad": 0, "w": 1e-5, "logits": 1e-5,
           "shared": 0, "route_ranks": 0.0}
    return all(errs[k] <= v for k, v in tol.items() if k in errs)


def route_check(x_f, w_r, bias_r, ids, wts):
    """Picked sets against torch (a row whose 8th and 9th keys are within 1e-5 is
    excused), the weights in id order, the shared slot."""
    s = torch.sigmoid(x_f.float() @ w_r.float().T)
    keys = s + bias_r.float()
    kv, ki = keys.topk(TOP_K + 1, dim=1)
    ref_ids = ki[:, :TOP_K]
    tie = (kv[:, TOP_K - 1] - kv[:, TOP_K]) < 1e-5
    mi = ids[:, :TOP_K].long()
    same = (mi.sort(1).values == ref_ids.sort(1).values).all(1)
    bad = int((~same & ~tie).sum().item())
    ws = s.gather(1, mi)
    w_ref = ws / ws.sum(1, keepdim=True) * 2.5
    shared = int(((ids[:, TOP_K] != N_EXPERTS) | (wts[:, TOP_K] != 1.0)).sum().item())
    return {"ids_bad": bad, "ties": int(tie.sum().item()), "w": rel(wts[:, :TOP_K], w_ref),
            "shared": shared}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", default="3,6,16")
    ap.add_argument("--tt", type=int, default=6)
    ap.add_argument("--state-bf16", default="1")
    ap.add_argument("--stop", default="route", choices=layer.STOPS)
    ap.add_argument("--launches", type=int, default=10)
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--oneshot-rows", type=int, default=0)
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args()
    rank = int(os.environ["RANK"])
    tp = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    dev = torch.device("cuda", rank)
    ok = True
    for reqs in map(int, a.reqs.split(",")):
        for sbf in map(int, a.state_bf16.split(",")):
            ok &= run(a, reqs, sbf, rank, tp, dev, cpu)
    flags = gather(torch.tensor([int(ok)], device=dev))
    if rank == 0:
        print("PASS" if all(f.item() for f in flags) else "FAIL", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
