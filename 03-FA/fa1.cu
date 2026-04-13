// an explainable version of tspeterkims flash attention kernel

#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// the host code that uses torch
torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int Bc = 32; // tile size of key/value 
    const int Br = 32; // tile size for query

    // the layout is (batch size, n heads, seq len, head dim)

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); // tiles need to cover the seq len
    const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q); // output 
    auto l = torch::zeros({B, nh, N}); // denominator (sum of exponentials) for softmax.
    auto m = torch::full({B, nh, N}, -INFINITY); // running max
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);

    // SRAM size needed per block
    // each q, k, v tile is Bc * d, and the intermediate tile is Bc * Br
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    fa1_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}