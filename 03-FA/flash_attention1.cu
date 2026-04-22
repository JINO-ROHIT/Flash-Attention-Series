#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>


__global__ void fa1(float* __restrict__ d_query, float* __restrict__ d_key,
                    float* __restrict__ d_value, float* __restrict__ d_output,
                    const int batch_size, const int n_head,
                    const int seq_len,   const int head_embd,
                    const float scale,   const int tile)
{
    extern __shared__ float sram[];
    int b = blockIdx.x;  
    int h = blockIdx.y;
    int t = threadIdx.x; // each thread owns one query row

    // one single head has seq len * head dim
    int head_size = seq_len * head_embd; 
    // conside you have 2 batches and 10 heads
    // if in batch 1, then you skip 1 * 10 heads = 10
    int head_offset = (b * n_head + h) * head_size;

    float* q_ptr = d_query  + head_offset;
    float* k_ptr = d_key    + head_offset;
    float* v_ptr = d_value  + head_offset;
    float* o_ptr = d_output + head_offset;

    float* k_tile = sram;
    float* v_tile = sram + tile * head_embd;

    // each thread loads q row into registers so we dont have to read from global mem
    // also think why we need only 64, 
    // since we launch seq len threads and head dim is 64, a query is seq len * head dim
    float q_reg[64];
    for (int d = 0; d < head_embd; d++)
        q_reg[d] = q_ptr[t * head_embd + d];

    float o_reg[64] = {};
    float m_i = -1e9f; // max for online softmax
    float l_i = 0.0f; // sum for online softmax

    // we iterate through the seq len of the K and V
    for (int tile_idx = 0; tile_idx < seq_len; tile_idx += tile) {
        int tile_elems = tile * head_embd;

        // we have 128 threads to load tile of size 64 * embed dim of size 64 = 4096 floats
        // so each thread should load 4096 / 128 = 32 floats
        // thread 0 loads 0, 127, etc
        // thread 1 loads 1, 128 etc so the memoery coelasced
        for (int i = t; i < tile_elems; i += blockDim.x) {
            int row = i / head_embd;
            int col = i % head_embd;
            int gr = tile_idx + row;
            k_tile[i] = k_ptr[gr * head_embd + col];
        }
        for (int i = t; i < tile_elems; i += blockDim.x) {
            int row = i / head_embd;
            int col = i % head_embd;
            int gr = tile_idx + row;
            v_tile[i] = v_ptr[gr * head_embd + col];
        }
        __syncthreads();

        float s[64]; 
        float m_tile = -1e9f; // max for a tile for computing safe softmax

        // do attention for this query
        for (int j = 0; j < tile; j++) {
            float dot = 0.0f;
            for (int d = 0; d < head_embd; d++)
                dot += q_reg[d] * k_tile[j * head_embd + d];
            s[j] = dot * scale;
            if (s[j] > m_tile) m_tile = s[j];
        }

        float m_new  = fmaxf(m_i, m_tile);
        float sc_old = expf(m_i - m_new);
        float l_new  = l_i * sc_old;
        for (int d = 0; d < head_embd; d++) o_reg[d] *= sc_old;

        for (int j = 0; j < tile; j++) {
            float p = expf(s[j] - m_new);
            l_new += p;
            for (int d = 0; d < head_embd; d++)
                o_reg[d] += p * v_tile[j * head_embd + d];
        }
        m_i = m_new; 
        l_i = l_new;
        __syncthreads();
    }
    for (int d = 0; d < head_embd; d++)
        o_ptr[t * head_embd + d] = o_reg[d] / l_i;
}


int main()
{
    // B - batch size, H - no of heads, S - seq len, D = head_dim
    const int B = 8, H = 12, S = 128, D = 64, TILE = 64;
    const int N = B * H * S * D;
    const float scale = 1.0f / sqrtf((float)D);

    float *hQ, *hK, *hV, *hO_gpu;
    hQ     = (float*)malloc(N * sizeof(float));
    hK     = (float*)malloc(N * sizeof(float));
    hV     = (float*)malloc(N * sizeof(float));
    hO_gpu = (float*)malloc(N * sizeof(float));

    srand(42);
    for (int i = 0; i < N; i++) {
        hQ[i] = ((float)rand()/RAND_MAX)*2.0f - 1.0f;
        hK[i] = ((float)rand()/RAND_MAX)*2.0f - 1.0f;
        hV[i] = ((float)rand()/RAND_MAX)*2.0f - 1.0f;
    }

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, N*sizeof(float)); cudaMalloc(&dK, N*sizeof(float));
    cudaMalloc(&dV, N*sizeof(float)); cudaMalloc(&dO, N*sizeof(float));
    cudaMemcpy(dQ, hQ, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, N*sizeof(float), cudaMemcpyHostToDevice);

    size_t sram = 2 * TILE * D * sizeof(float);
    dim3 block(S);
    dim3 grid(B, H);

    // warm-up
    fa1<<<grid, block, sram>>>(dQ, dK, dV, dO, B, H, S, D, scale, TILE);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int REPS = 100;
    cudaEventRecord(t0);
    for (int r = 0; r < REPS; r++)
        fa1<<<grid, block, sram>>>(dQ, dK, dV, dO, B, H, S, D, scale, TILE);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms, t0, t1);
    gpu_ms /= REPS;

    cudaMemcpy(hO_gpu, dO, N*sizeof(float), cudaMemcpyDeviceToHost);

    printf("  GPU time  : %.3f ms  (avg over %d runs)\n", gpu_ms, REPS);

    free(hQ); free(hK); free(hV); free(hO_gpu);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}