# SPDX-License-Identifier: Apache-2.0
"""Stage timeline of the wide launch milestone 1a (``m1a.build(..., trace=True)``):
per stage, when its first CTA starts and its last CTA finishes (us from launch start).

    python m1a_trace.py [--reqs 3,16] [--tt 6]
"""

import argparse
import os

import torch

os.environ.setdefault("SGLANG_USE_AITER", "1")

from sgl_amd_mono.common.plan import BLOCKS  # noqa: E402
from sgl_amd_mono.glm53_flash.gfx950.config import HIDDEN  # noqa: E402
from sgl_amd_mono.glm53_flash.gfx950.wide import kda, m1a, mhc  # noqa: E402

HC, NH, KD, P, NPROJ = mhc.HC, kda.NH, kda.KD, kda.P, kda.NPROJ
NAMES = ["start", "aseam", "agate", "-", "inproj", "kda", "epoch", "proj wait",
         "k:wait", "k:fg+loads", "k:conv+l2", "k:recur", "k:norm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", default="3,16")
    ap.add_argument("--tt", type=int, default=6)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    dev, tt = "cuda", a.tt
    torch.manual_seed(0)
    for reqs in map(int, a.reqs.split(",")):
        m, slots, L = reqs * tt, reqs + 2, tt
        t = lambda *s, dt=torch.bfloat16: (torch.randn(*s, device=dev) * 0.05).to(dt)  # noqa: E731
        res_in, fn = t(m, HC, HIDDEN), t(24, HC * HIDDEN, dt=torch.float32)
        hc_scale, hc_base = torch.tensor([0.8, 0.6, 1.1], device=dev), t(24, dt=torch.float32)
        norm_w, x = t(HIDDEN) + 1, torch.zeros(m, HIDDEN, device=dev, dtype=torch.bfloat16)
        post_out, comb_out = torch.zeros(m, HC, device=dev), torch.zeros(m, HC * HC, device=dev)
        w_in, proj = t(NPROJ, HIDDEN), torch.zeros(m, NPROJ, device=dev, dtype=torch.bfloat16)
        conv_w, conv_st = t(3 * P, 4, dt=torch.float32), t(slots, 3, 3 * P).transpose(1, 2)
        idx = torch.arange(1, reqs + 1, device=dev, dtype=torch.int32)
        iw = torch.zeros(slots, tt, 3, 3 * P, device=dev, dtype=torch.bfloat16).transpose(2, 3)
        w_fgb, a_log, dt_bias = t(2, P, KD), torch.zeros(NH, device=dev), t(P, dt=torch.float32)
        state = t(slots, NH, KD, KD)
        rawv, rawk = t(slots, NH, L, KD), t(slots, NH, L, KD)
        ring_g, ring_beta = t(slots, NH, L, KD, dt=torch.float32), t(slots, NH, L, dt=torch.float32)
        on_w, core = t(KD, dt=torch.float32) + 1, torch.zeros(m, P, device=dev, dtype=torch.bfloat16)
        launch = m1a.build(m, tt, -5.0, 1e-5, True, True, trace=True)
        plan = launch.plan
        sc = m1a.scratch_layout(m, True)
        epoch = torch.zeros(plan["epoch_words"], device=dev, dtype=torch.int32)
        scratch = torch.zeros(plan["scratch_words"], device=dev, dtype=torch.int32)
        ptrs = (res_in, fn, hc_scale, hc_base, norm_w, x, post_out, comb_out, w_in, proj, conv_w,
                conv_st, idx, iw, idx, w_fgb, a_log, dt_bias, state, idx, rawv, rawk, ring_g,
                ring_beta, idx, on_w, core, epoch, scratch)
        ints = (conv_st.stride(0), conv_st.stride(2), iw.stride(0), iw.stride(1), iw.stride(3),
                rawv.stride(0), rawk.stride(0), ring_g.stride(0), ring_beta.stride(0), L)
        pl = [p.data_ptr() for p in ptrs]
        acc = None
        for _ in range(a.reps):
            launch(*pl, *ints, stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
            tr = scratch[sc["trace"]:].view(torch.int64).view(BLOCKS, m1a.TRACE_POINTS).double()
            tr = (tr - tr[:, 0].min()) / 100.0  # 100 MHz -> us
            acc = tr if acc is None else acc + tr
        tr = acc / a.reps
        g = plan["gemm"]
        print(f"rows {m}  tasks aseam {plan['n_aseam']} inproj {plan['n_inproj']} "
              f"({g.bm}x{g.bn} ks{g.ks}) kda {plan['n_kda']}")
        for p in (0, 1, 2, 4, 7, 8, 9, 10, 11, 12, 5, 6):
            col = tr[:, p]
            col = col[(col > -1e5) & (col < 1e5) & (col != 0)]
            print(f"  {NAMES[p]:>10}  min {col.min():6.2f}  median {col.median():6.2f}  max {col.max():6.2f} us")


if __name__ == "__main__":
    main()
