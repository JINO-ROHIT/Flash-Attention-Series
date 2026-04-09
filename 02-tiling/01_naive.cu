#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// we are doing A = M x K 
//              B = K x N

__global__ void matmul(float* A, float* B, float* C, const int M, const int K, const int N){
    int t_row = blockIdx.y * blockDim.y + threadIdx.y;
    int t_col = blockIdx.x * blockDim.x + threadIdx.x;

    if(t_row < M && t_col < N){
        float tmp = 0.0f;
        for(int i = 0; i < K; i++){
            tmp += A[t_row * K + i] * B[i * N + t_col];
        }

        C[t_row * N + t_col] = tmp;
    }
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
    
    matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    //cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}