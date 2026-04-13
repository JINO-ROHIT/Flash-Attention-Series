### implementing flash attention 1

paper - https://arxiv.org/pdf/2205.14135

once we understand the important concepts of online softmax and tiling, we can now apply those to implementing IO aware flash attention.

we will implement this in CUDA.


reference
this kernel is taken from https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu
