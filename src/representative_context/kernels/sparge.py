"""FP32 aggregation with the official selected block pattern."""

import triton
import triton.language as tl


@triton.jit
def sparse_block_attention(
    Q,
    K,
    V,
    LUT,
    COUNTS,
    OUT,
    sqb,
    sqh,
    sqn,
    skb,
    skh,
    skn,
    svb,
    svh,
    svn,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    HEADS: tl.constexpr,
    QBLOCKS: tl.constexpr,
    KBLOCKS: tl.constexpr,
    DIM: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    tile, head, batch = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    rows = tile * BM + tl.arange(0, BM)
    dims = tl.arange(0, DIM)
    cols = tl.arange(0, BN)
    q = tl.load(
        Q + batch * sqb + head * sqh + rows[:, None] * sqn + dims[None, :],
        mask=rows[:, None] < NQ,
        other=0,
    )
    row = tile * BM // 128
    count = tl.load(COUNTS + (batch * HEADS + head) * QBLOCKS + row)
    base = ((batch * HEADS + head) * QBLOCKS + row) * KBLOCKS
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, DIM), tl.float32)
    block = 0
    for i in range(count):
        block += tl.load(LUT + base + i)
        keys = block * BN + cols
        key = tl.load(
            K + batch * skb + head * skh + keys[None, :] * skn + dims[:, None],
            mask=keys[None, :] < NK,
            other=0,
        )
        score = tl.dot(q, key, input_precision="tf32x3") * SCALE
        score = tl.where(keys[None, :] < NK, score, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        weights = tl.exp(score - new_max[:, None])
        alpha = tl.exp(maximum - new_max)
        normalizer = normalizer * alpha + tl.sum(weights, 1)
        value = tl.load(
            V + batch * svb + head * svh + keys[:, None] * svn + dims[None, :],
            mask=keys[:, None] < NK,
            other=0,
        )
        acc = acc * alpha[:, None] + tl.dot(weights, value, input_precision="tf32x3")
        maximum = new_max
    result = acc / normalizer[:, None]
    tl.store(
        OUT + ((batch * HEADS + head) * NQ + rows[:, None]) * DIM + dims[None, :],
        result,
        mask=rows[:, None] < NQ,
    )
