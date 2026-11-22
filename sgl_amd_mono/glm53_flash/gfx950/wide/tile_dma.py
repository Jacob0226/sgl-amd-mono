# SPDX-License-Identifier: Apache-2.0
"""bf16 MFMA tile GEMM stage fed by direct-to-LDS DMA (the wide launch's dense GEMMs).

The pipeline follows aiter's FlyDSL hgemm (``gemm_a16w16_gfx950``, the kernel tgemm
picks for GLM-5.3-Flash's in_proj / o_proj at 16-128 rows): x and W tiles go
global -> LDS with ``buffer_load ... lds`` (no VGPR staging) into a ring of
``stages`` K steps, grouped by ``asyncmark``; a step is computed once its group
lands. LDS rows are BK bf16 with no padding; the 16 B piece index is XORed with
bits 3..5 of the row's piece number (aiter's Swizzle<3,3,3>), so the 16 rows of a
ds_read_b128 land on distinct banks. The DMA writes 64 lanes x 16 B contiguous
LDS per wave, so the swizzle is applied on the global side: the lane that owns
LDS piece g loads logical piece (g % P) ^ ((g >> 3) & 7) of its row.

A task is a [bm token rows x bn weight rows] tile over K columns k0 .. k0 + kt;
the 8 waves form a wm x wn grid of sub-tiles. With ks > 1 a task's partial goes
out through ``emit`` / ``lin`` as in ``kimi_k31.gfx950.stages.tile_mma``.

``w8``: W is FP8 E4M3 with one f32 scale per row (``pack_w8``: the scales follow the
[n, k] bytes in the same buffer). Its LDS rows are bk bytes; a fragment's 8 bytes
widen to bf16 exactly, and the row scale is applied to the f32 accumulator.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64

import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import CM_NT, fp8_to_bf16x8, mfma_bf16, rsrc, uniform
from sgl_amd_mono.common.plan import THREADS, WAVES, kernel_sources, key_tuple
from sgl_amd_mono.kimi_k31.gfx950.stages.tile import (
    atomic_emit,
    atomic_lin,
    bf16_emit,
    f32_emit,
)
from sgl_amd_mono.kimi_k31.gfx950.stages.tile_mma import _shuffle

SOURCES = kernel_sources(__name__)

STAGES = 4
DMA_BYTES = 16


def lds_words(bm, bn, bk=128, stages=STAGES, w8=False):
    """i32 words of LDS a task uses: the ring of x and W tiles (the shuffled
    epilogue's f32 [bm, bn + 4] tile reuses it)."""
    return max(stages * (bm * bk // 2 + bn * bk // (4 if w8 else 2)), bm * (bn + 4))


def pack_w8(w):
    """bf16 [n, k] -> one uint8 buffer: FP8 E4M3 [n, k] (row absmax / 448 scales),
    then the n f32 scales."""
    n, k = w.shape
    assert (n * k) % 16 == 0
    s = (w.float().abs().amax(1) / 448.0).clamp_min(1e-12)
    out = torch.empty(n * k + n * 4, dtype=torch.uint8, device=w.device)
    out[: n * k].view(torch.float8_e4m3fn).view(n, k).copy_(
        (w.float() / s[:, None]).to(torch.float8_e4m3fn)
    )
    out[n * k :].view(torch.float32).copy_(s)
    return out


def unpack_w8(buf, n, k):
    """``pack_w8``'s buffer -> the f32 [n, k] it stands for."""
    q = buf[: n * k].view(torch.float8_e4m3fn).view(n, k).float()
    return q * buf[n * k :].view(torch.float32)[:, None]


def _phys(row, lp, pieces):
    """LDS piece of logical 16 B piece ``lp`` in ``row`` (self-inverse)."""
    return lp ^ (((row * pieces + lp) >> 3) & 7)


def _dma(tid, src, ld, lim, base, k0, rows, bk, lds, aux, rmap=None, esz=2):
    """Issue one K step's [rows, bk] tile of ``src`` (rows base .., clamped to
    lim - 1; ``esz``-byte elements) into ``lds`` (i32 words): 64 pieces a wave per round."""
    pieces = bk * esz // 16
    tot = rows * pieces
    assert tot % 64 == 0
    lane, wave = tid % 64, tid // 64
    for i in range_constexpr((tot + THREADS - 1) // THREADS):
        live = min(WAVES, (tot - i * THREADS) // 64)
        # a wave past the tile repeats the last live wave's pieces (same data, same words)
        wv = wave if const_expr(live == WAVES) else fx.min(wave, live - 1)
        g = i * THREADS + wv * 64 + lane
        lp = (g % pieces) ^ ((g >> 3) & 7)
        row = fx.min(base + g // pieces, lim - 1)
        if const_expr(rmap is not None):
            row = rmap(row)
        voff = (fx.Int32(row) * ld + k0) * esz + lp * 16
        dst = lds + uniform(fx.Int32(i * THREADS * 4) + wv * 256)
        rocdl.raw_ptr_buffer_load_async_lds(
            src,
            fx.to_llvm_ptr(dst),
            fx.Int32(DMA_BYTES).ir_value(),
            fx.Int32(voff).ir_value(),
            fx.Int32(0).ir_value(),
            fx.Int32(0).ir_value(),
            aux=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), aux),
        )


def _frag(lds, rows_off, row, lp, pieces):
    return fx.Vector(
        fx.ptr_load(
            lds + rows_off + row * (pieces * 4) + _phys(row, lp, pieces) * 4,
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )


def dma_tile(
    tid,
    x,
    ldx,
    m,
    m0,
    w,
    ldw,
    n,
    n0,
    k0,
    kt,
    bm,
    bn,
    wm,
    wn,
    ab,
    emit,
    bk=128,
    stages=STAGES,
    lin=None,
    cl=None,
    xcm=0,
    wrow=None,
    wait_x=None,
    w8=False,
):
    """One task: K columns k0 .. k0 + kt of the [bm, bn] tile at (m0, n0) through
    the LDS ring ``ab`` (``lds_words(bm, bn, bk, stages)`` i32 words). ``emit(row,
    col, v, live)`` gets 4 output columns n0 + col .. of token row ``row``; with
    ``lin`` the tile is shuffled through the f32 LDS ``cl`` (aliasing ``ab``) into
    lane-contiguous adds. ``xcm``: x's cache policy; W loads are nontemporal.
    ``wait_x()``: x is not ready yet: the first stages - 1 W slabs go out before it
    returns (their own asyncmark groups, ahead of the x slabs': a step still waits
    for all but the last stages - 2 groups)."""
    assert wm * wn == WAVES and kt % bk == 0 and bk in (64, 128)
    # the swizzle needs 8 pieces a row: an FP8 row of bk = 64 has 4
    assert not w8 or bk == 128
    lane, wave = tid % 64, tid // 64
    tm, tn = bm // wm, bn // wn
    mt, nt = tm // 16, tn // 16
    assert mt * 16 == tm and nt * 16 == tn
    wy, wx = wave // wn, wave % wn
    pieces = bk // 8
    wpieces = bk // 16 if w8 else pieces
    rw = bk // 2
    xr, wr = rsrc(x), rsrc(w)
    steps = kt // bk
    xw, ww = bm * rw, bn * wpieces * 4
    sw = xw + ww
    acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(nt)] for _ in range(mt)]

    def load_w(j, s):
        _dma(
            tid, wr, ldw, n, n0, k0 + j * bk, bn, bk, ab + s * sw + xw, CM_NT, wrow,
            1 if w8 else 2,
        )

    def w_frag(s, row, c):
        """W row ``row``'s K chunk ``c`` (8 values) of stage ``s`` as 8 bf16."""
        if const_expr(not w8):
            return _frag(ab, s * sw + xw, row, c, pieces).bitcast(fx.BFloat16)
        off = s * sw + xw + row * (wpieces * 4) + _phys(row, c // 2, wpieces) * 4 + (c % 2) * 2
        v = fx.Vector(fx.ptr_load(ab + off, result_type=fx.Vector.make_type(2, fx.Int32)))
        return fp8_to_bf16x8(v[0], v[1])

    def load_x(j, s):
        _dma(tid, xr, ldx, m, m0, k0 + j * bk, bm, bk, ab + s * sw, xcm)

    def load(j, s):
        load_w(j, s)
        load_x(j, s)

    def compute(s):
        for kk in range_constexpr(bk // 32):
            a = [w_frag(s, wx * tn + b * 16 + lane % 16, kk * 4 + lane // 16) for b in range(nt)]
            for t in range_constexpr(mt):
                xb = _frag(
                    ab, s * sw, wy * tm + t * 16 + lane % 16, kk * 4 + lane // 16, pieces
                ).bitcast(fx.BFloat16)
                for b in range_constexpr(nt):
                    acc[t][b] = mfma_bf16(a[b], xb, acc[t][b])

    pre = min(stages - 1, steps)
    if const_expr(wait_x is None):
        for j in range_constexpr(pre):
            load(j, j)
            rocdl.asyncmark()
    else:
        for j in range_constexpr(pre):
            load_w(j, j)
            rocdl.asyncmark()
        wait_x()
        for j in range_constexpr(pre):
            load_x(j, j)
            rocdl.asyncmark()
    rocdl.sched_barrier(0)
    for j in range_constexpr(steps):
        ahead = j + stages - 1
        rocdl.wait_asyncmark(min(stages - 2, steps - 1 - j))
        rocdl.s_barrier()
        if const_expr(ahead < steps):
            load(ahead, ahead % stages)
            rocdl.asyncmark()
        compute(j % stages)
        rocdl.sched_barrier(0)
    if const_expr(w8):
        # an accumulator holds W rows n0 + col .. + 3 of one token row
        r_s = rsrc(w + fx.Int64(n * ldw), n * 4)
        for b in range_constexpr(nt):
            col = wx * tn + b * 16 + 4 * (lane // 16)
            sv = fx.Vector(bo.buffer_load(r_s, n0 + col, vec_width=4, dtype=T.f32))
            for t in range_constexpr(mt):
                acc[t][b] = fx.Vector.from_elements(
                    [acc[t][b][i] * sv[i] for i in range(4)], fx.Float32
                )
    if const_expr(lin is not None):
        _shuffle(tid, lane, wave, acc, cl, bm, bn, tm, tn, wy, wx, m0, lin)
        return
    for t in range_constexpr(mt):
        for b in range_constexpr(nt):
            row = m0 + wy * tm + t * 16 + lane % 16
            col = wx * tn + b * 16 + 4 * (lane // 16)
            emit(row, col, acc[t][b], None)


_BUILDS: dict = {}


def standalone(
    m, n, k, bm=32, bn=128, wm=2, wn=4, ks=4, bk=128, stages=STAGES, mode="linbf",
    w8=False,
):
    """Grid (n tiles, ks, m tiles): bf16 out[m, n] at ks = 1; with ks > 1 by
    ``mode`` as in ``tile_mma.standalone`` (part / atom / atombf / lin / linbf)."""
    keyed = key_tuple(locals(), SOURCES)
    key = (m, n, k, bm, bn, wm, wn, ks, bk, stages, mode, w8)
    if key in _BUILDS:
        return _BUILDS[key]
    kt = k // ks
    assert k % ks == 0 and kt % bk == 0
    tn_, tm_ = (n + bn - 1) // bn, (m + bm - 1) // bm
    words = stages * (bm * bk // 2 + bn * bk // (4 if w8 else 2))

    @fx.struct
    class AbLds:
        ab: fx.Array[fx.Int32, words, 16]

    @fx.struct
    class CLds:
        c: fx.Array[fx.Float32, bm * (bn + 4), 16]

    @fx.union
    class Smem:
        ab: AbLds
        c: CLds

    name = f"g53w_dma_m{m}_n{n}_k{k}_t{bm}x{bn}x{bk}x{stages}_w{wm}x{wn}_ks{ks}_{mode}{'_w8' if w8 else ''}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dma_kernel(x: Int64, w: Int64, out: Int64):
        tid = fx.thread_idx.x
        n0 = fx.block_idx.x * bn
        sk = fx.block_idx.y
        m0 = fx.block_idx.z * bm
        st = fx.SharedAllocator().allocate(Smem)
        ab, cl = st.ab.peek().ab.ptr, st.c.peek().c.ptr
        lin = None
        if const_expr(ks == 1):
            emit = bf16_emit(out, n, n0, m, n)
        elif const_expr(mode == "part"):
            emit = f32_emit(out + fx.Int64(sk) * (m * n * 4), n, n0, m, n)
        elif const_expr(mode in ("atom", "atombf")):
            emit = atomic_emit(out, n, n0, m, n, mode == "atombf")
        else:
            emit = None
            lin = atomic_lin(out, n, n0, m, n, mode == "linbf")
        dma_tile(
            tid, x, k, m, m0, w, k, n, n0, sk * kt, kt,
            bm, bn, wm, wn, ab, emit, bk, stages, lin, cl, w8=w8,
        )

    @flyc.jit
    def launch(x: Int64, w: Int64, out: Int64, stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        _ = keyed
        dma_kernel(x, w, out).launch(
            grid=(tn_, ks, tm_), block=(THREADS,), stream=stream
        )

    _BUILDS[key] = launch
    return launch
