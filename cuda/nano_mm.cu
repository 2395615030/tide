#include <cuda_runtime.h>
#include <stdio.h>

// C[b,t,d] = sum_{k<=t} A[t,k] * B[b,k,d]   （因果 batched matmul）
__global__ void mm_causal_kernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int Bn, int T, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Bn * T * D;
    if (idx >= total) return;
    int d = idx % D;
    int t = (idx / D) % T;
    int b = idx / (D * T);
    const float* a = A + (long)t * T;
    const float* bb = B + (long)b * T * D;
    float acc = 0.f;
    for (int k = 0; k <= t; ++k) acc += a[k] * bb[(long)k * D + d];
    C[idx] = acc;
}

extern "C" __declspec(dllexport)
int mm_causal(const float* hA, const float* hB, float* hC, int Bn, int T, int D) {
    float *dA, *dB, *dC;
    size_t na = (size_t)T * T, nb = (size_t)Bn * T * D, nc = nb;
    cudaMalloc(&dA, na * 4);
    cudaMalloc(&dB, nb * 4);
    cudaMalloc(&dC, nc * 4);
    cudaMemcpy(dA, hA, na * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, nb * 4, cudaMemcpyHostToDevice);
    int total = (int)nc;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mm_causal_kernel<<<blocks, threads>>>(dA, dB, dC, Bn, T, D);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    cudaMemcpy(hC, dC, nc * 4, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

extern "C" __declspec(dllexport)
int gpu_info(int* sm_count, int* mem_mb, char* name, int name_len) {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return -1;
    *sm_count = p.multiProcessorCount;
    *mem_mb = (int)(p.totalGlobalMem / (1024 * 1024));
    snprintf(name, name_len, "%s", p.name);
    return 0;
}
