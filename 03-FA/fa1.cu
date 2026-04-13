// rough outline for what we wanna do

// we have Q, K, V
// O matrix for output
// Bc(tile for K/V) and Br(tile for Q)

// for each query, we need
    // m[i] = -infinity   # running max
    // l[i] = 0.0         # running denominator sum


//load tile K and V
    // load tile of Q
        // for each query row in this tile
            // S = Qi[r] * Kj^T  (dot products with all key rows in tile) ---
        
            // do the online softmax thingy



// Q, K, V tensors of shape (batch, seq len, n heads, head dim)
// O tensor of shape (batch, seq len, n heads, head dim)

template <typename T>
__global__ void fa1(const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V, T* __restrict__ O,
    int batch_size, int target_seq_len, int src_seq_len, int n_heads, int head_dim, float scale, bool is_causal){
        
}


// host code
#include <cuda_runtime.h>
#include <vector>

template <typename T>
void flashAttention(
    const std::vector<T>& h_q,   // [batch, seq_len, num_heads, head_dim]
    const std::vector<T>& h_k,   // [batch, seq_len, num_heads, head_dim]
    const std::vector<T>& h_v,   // [batch, seq_len, num_heads, head_dim]
    std::vector<T>&       h_o,   // [batch, seq_len, num_heads, head_dim]
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    bool is_causal)
{

    const size_t total = (size_t)batch_size * seq_len * num_heads * head_dim;

    h_o.resize(total);

    T *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, total * sizeof(T));
    cudaMalloc(&d_k, total * sizeof(T));
    cudaMalloc(&d_v, total * sizeof(T));
    cudaMalloc(&d_o, total * sizeof(T));

    cudaMemcpy(d_q, h_q.data(), total * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), total * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), total * sizeof(T), cudaMemcpyHostToDevice);

    //  1 warp  (32 threads) handles 1 token
    //  1 block (8 warps)    handles 8 consecutive tokens
    //
    //  grid.x = enough blocks to cover all seq_len tokens
    //  grid.y = one slice per attention head
    //  grid.z = one slice per batch element

    const int warps_per_block   = 8;
    const int threads_per_block = warps_per_block * 32;   // 256

    dim3 grid((seq_len + warps_per_block - 1) / warps_per_block, // grid x
              num_heads, // grid y
              batch_size); // grid z
    dim3 block(threads_per_block);

    const float scale = 1.0f / std::sqrt((float)head_dim);


    fa1<T><<<grid, block>>>(
        d_q, d_k, d_v, d_o,
        batch_size, seq_len, num_heads, head_dim,
        scale, is_causal);

    cudaDeviceSynchronize();

    cudaMemcpy(h_o.data(), d_o, total * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}


int main() {
    const int batch_size = 2;
    const int seq_len    = 128;
    const int num_heads  = 8;
    const int head_dim   = 64;
    const bool is_causal = true;

    const size_t total = batch_size * seq_len * num_heads * head_dim;

    std::vector<float> h_q(total), h_k(total), h_v(total), h_o;
    for (auto& x : h_q) x = (float)rand() / RAND_MAX - 0.5f;
    for (auto& x : h_k) x = (float)rand() / RAND_MAX - 0.5f;
    for (auto& x : h_v) x = (float)rand() / RAND_MAX - 0.5f;

    flashAttention<float>(h_q, h_k, h_v, h_o,
                          batch_size, seq_len, num_heads, head_dim, is_causal);
    return 0;
}