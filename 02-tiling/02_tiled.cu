#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// we are doing A = M x K 
//              B = K x N

#define BM 16
#define BK 16
#define BN 16

__global__ void tiled_matmul(float* A, float* B, float* C, const int M, const int K, const int N){
    __shared__ float As[BM * BK]; // 256 elements can be loaded into this
    __shared__ float Bs[BK * BN];


    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;

    int tile_row = threadIdx.y;
    int tile_col = threadIdx.x;

    float tmp = 0.0f;
    for(int tile_idx = 0; tile_idx < K/BK; tile_idx += 1){
        // first we load onto tiles A and B
        // As[tile_row * BK + tile_col] = A[global_row * K + tile_idx * BK + tile_col];
        // Bs[tile_row * BN + tile_col] = B[(tile_idx * BK + tile_row) * N + global_col];

        As[tile_row * BK + tile_col] = A[global_row * K + tile_idx * BK + tile_col];
        Bs[tile_row * BN + tile_col] = B[(tile_idx * BK + tile_row) * N + global_col];
        __syncthreads();
        
        for(int i = 0; i < BK; i++){
            tmp += As[tile_row * BK + i] * Bs[i * BN + tile_col];
        }

        __syncthreads();    
    }
    C[global_row * N + global_col] = tmp;
}

int main(){
    const int M = 4096, N = 4096, K = 4096;

    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *h_C;

    h_A = (float*)malloc(M * K * sizeof(float));
    h_B = (float*)malloc(K * N * sizeof(float));
    h_C = (float*)malloc(M * N * sizeof(float));
    

    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;
    

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    int BLOCKSIZE = 16;
    dim3 gridDim(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE));
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    
    tiled_matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    //cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}