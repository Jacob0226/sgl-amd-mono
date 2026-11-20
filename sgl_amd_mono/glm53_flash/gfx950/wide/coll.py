# SPDX-License-Identifier: Apache-2.0
"""Two-shot all-reduce of the wide launch's o_proj partial [rows, HIDDEN] over ``tp``
ranks, in peer memory (``common.peer.PeerBuffer``, two parity halves).

Shot 1 (RS): the o_proj tile epilogue (Kimi's ``_push_band_flag``) pushes a band of
rb rows x bn columns of this rank's partial into the column owner's ``rs`` region
[src][rows][sl] (sl = HIDDEN / tp), then flags (src, piece) in the owner's ``fl``.
Shot 2 (AG): the owner sums a piece's tp sources in rank order (f32, then bf16, as
the custom all-reduce does) and stores the sum into every rank's ``ag`` region
[rows][HIDDEN], then flags (owner, piece) in every rank's ``agf``. Every rank thus
holds the same bits.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import CM_SYS, bf2_f32, bf16_pair, rsrc, uniform
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES
from sgl_amd_mono.glm53_flash.gfx950.config import HIDDEN


def pieces(rows, rb, bn, tp):
    """An owner's pieces: row bands of rb x column tiles of bn in its slice."""
    return -(-rows // rb) * ((HIDDEN // tp) // bn)


def layout(rows, rb, bn, tp):
    """Byte offsets of one parity half and the half's size."""
    sl = HIDDEN // tp
    rs = tp * rows * sl * 2
    ag = rows * HIDDEN * 2
    fl = tp * BLOCKS * 8
    agf = tp * pieces(rows, rb, bn, tp) * 8
    offs = {"rs": 0, "ag": rs, "fl": rs + ag, "agf": rs + ag + fl}
    half = (rs + ag + fl + agf + 255) // 256 * 256
    return offs, half


def peer_bytes(rows, rb, bn, tp):
    return 2 * layout(rows, rb, bn, tp)[1] + guard.FLAG_BYTES


@flyc.jit
def put_each(c, tp, region, i, v):
    """Lane 0 of wave p puts (i, v) into rank p's ``region`` (p < tp)."""
    assert WAVES >= tp
    tid = c["tid"]
    p = uniform(tid // 64)
    if (p < tp) & (tid % 64 == 0):
        c["put"](c[f"{region}_of"](p), i, v)


@flyc.jit
def reduce_bcast(c, piece, m, rb, bn, tp):
    """Owner side of piece ``piece``: wait for the tp sources, sum in rank order,
    store the bf16 sum into every rank's ``ag``, then flag it there."""
    tid = c["tid"]
    sl = HIDDEN // tp
    q8 = bn // 8
    ct = sl // bn
    if tid < tp:
        c["poll"]([(c["fl_own"], tid * BLOCKS + piece, 1)])
    gpu.barrier()
    band, cj = piece // ct, piece % ct
    own = c["rank"]
    r_rs = rsrc(c["rs_own"].value)
    dsts = [rsrc(c["ag_of"](p).value) for p in range(tp)]
    for u in range(tid, rb * q8, THREADS):
        row = band * rb + u // q8
        col = cj * bn + 8 * (u % q8)
        if row < m:
            got = [
                fx.Vector(
                    bo.buffer_load(
                        r_rs,
                        (src * m * sl + row * sl + col) // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_SYS,
                    )
                )
                for src in range(tp)
            ]
            acc = None
            for src in range_constexpr(tp):
                cur = [v for e in range(4) for v in bf2_f32(got[src][e])]
                acc = cur if acc is None else [a + b for a, b in zip(acc, cur)]
            words = fx.Vector.from_elements(
                [bf16_pair(acc[2 * j], acc[2 * j + 1]).bitcast(fx.Int32) for j in range(4)],
                fx.Int32,
            )
            for p in range_constexpr(tp):
                bo.buffer_store(
                    words, dsts[p], (row * HIDDEN + own * sl + col) // 2, cache_modifier=CM_SYS
                )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    put_each(c, tp, "agf", own * pieces(m, rb, bn, tp) + piece, fx.Int32(1))
