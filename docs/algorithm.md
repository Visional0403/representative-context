# Capacity and spatial sampling

For a layer with query stride `s`, additional KV reduction ratio `r`, and
actual KV token count `K`, the integer capacity is

```text
B = max(1, ceil(2**k / (s*r*r)))
G = ceil(K / B)
```

`B` bounds group capacity. Balanced groups contain `floor(K/G)` or
`ceil(K/G)` tokens. Changing image size changes the measured token count
and group count while retaining a fixed per-query capacity for a given
layer and `k`. The implementation computes the ceilings with integer
arithmetic, including non-divisible dimensions.

The default layout uses a 4 by 8 grid of sampling regions. Positions
within a region follow Morton order. On regular grids, a seed-dependent
coprime affine permutation spreads positions across groups. On irregular
grids, exact quotas allocate positions across regions and balance group
sizes. Low capacities can sample fewer than all 32 regions; the capacity
limit continues to hold. The layout seed for stage `j`, block `i` is
`seed + 100*j + i`, with zero-based indices.

Each KV token has a unique rank. `rank % G` is its group; integer division
by `G` gives its position within that group. Query positions inherit the
group of their nearest aligned KV grid position. This also defines
membership when query and KV dimensions have non-integer ratios.

The indexed kernel reads Q/K/V from their original addresses, performs
online softmax over retained KV positions, and writes outputs in the
original query order. Query and KV tiles are both 64; head channels are
masked to their true dimension. Supported head widths are 1 through 128.
FP32 dot products use Triton's `tf32x3` mode. Wider heads may require
`num_stages=1` to fit the GPU's per-block shared-memory limit; the
SegFormer-B0 configuration uses 32-wide heads and two pipeline stages.
Fixed index layouts are
cached outside steady-state timing.

One group retains all interactions and uses the optimized dense SDPA
path. Dense and sparse SRA share the projection, SR, normalization and
output projection weights. Dense FP32 execution selects PyTorch's
efficient-attention backend; half precision selects its Flash backend.
The SRA adapter supports batch-one inference.

The package also contains equal-offset Grid, balanced random and
region-clustered controls. They use the same rank representation and
indexed execution kernel. Grid requires dimensions compatible with its
rectangular sampling grid; clustered controls require divisible 2 by 2
tiles. These controls preserve exact group balance.
