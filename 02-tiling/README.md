### flash attention series part 2 - matrix multiplication

matrix multiplication is at the heart of the entire attention operation, so its important to be able to understand this very well. we also need to understand the different memory layers in the gpu and why tiling makes sense for doing flash attention.

$$C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}$$

we will be doing all of this in `CUDA C++` and use nsight for profiling.


#### profiler

to use nsight compute with UI, do -

```
sudo -E ncu-ui
```

to profile using command line, do -

```
sudo ncu <exe>
```

to get the ptx, do -

```
nvcc -ptx <kernel> -o <out.ptx>
```


1. `01_naive.cu` - a naive version for matrix multiplication.

```
memory throughput - 80% (l1 + l2 + dram) utilization high, 
duration - 120.83 ms
l1 texture cache throughput - 80.66%
l2 cache throughput - 20.04%

LSU. the LSU pipeline issues load, store, atomic, and reduction instructions to the L1TEX unit for global, local, and shared memory. 
LSU - 87% the highest means it spends all its time to fetch/move data.
FMA - 11%, fused mutiply add
ALU - 4% , performs bit level instructions
```

so our kernels seems more memory bound.

2. `02_tiled.cu` - matrix multiplication using  16x16 tiling.

```
memory throughput - 88.41% (l1 + l2 + dram) utilization very high, memory bound, ours blocks are waiting on data.
duration - 92.83 ms
l1 texture cache throughput - 88.44%
l2 cache throughput - 19.35% we hit l2 less compared to naive, but not by much

LSU. the LSU pipeline issues load, store, atomic, and reduction instructions to the L1TEX unit for global, local, and shared memory. 
LSU - 87% the highest means it spends all its time to fetch/move data.
FMA - 7.8%, fused mutiply add
ALU - 3.7% , performs bit level instructions
```