#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float alpha, int N) {
    // Write code here
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx>N) {
        return;
    }
    if (input[idx]>=0.0) {
        output[idx] =input[idx];
    } else {
        output[idx] =alpha*input[idx];
    }
    
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}