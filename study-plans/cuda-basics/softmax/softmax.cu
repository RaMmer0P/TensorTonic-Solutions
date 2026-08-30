#include <cuda_runtime.h>
#include <float.h>
__global__ void max_val (const float* in, float* out, int N) {
    __shared__ float s[256];
    // reduce function
    int idx = threadIdx.x;
    float thread_max = -FLT_MAX;
    // for every thread loop until we reach end by jumping block dimension each
    for (int i=idx;i<N;i+=blockDim.x) {
        thread_max = fmaxf(thread_max,in[i]);
    }
    s[idx] = thread_max;
    __syncthreads();
    for (int st = blockDim.x/2;st>0;st=st/2) {
        // reduce func to get the min over different threads
        if (idx<st) s[idx] = fmax(s[idx],s[st+idx]);
        __syncthreads();
    }
    if (idx==0) *out = s[0];
}

__global__ void sum_exp (const float* in, float* out, int N, float*d_max,float* sum) {
    __shared__ float s[256];
    // reduce function
    int idx = threadIdx.x;
    float thread_sum = 0;
    // for every thread loop until we reach end by jumping block dimension each
    for (int i=idx;i<N;i+=blockDim.x) {
        float e = expf(in[i]-*d_max);
        out[i] = e;
        thread_sum+= e;
    }
    s[idx] = thread_sum;
    __syncthreads();
    for (int st = blockDim.x/2;st>0;st=st/2) {
        // reduce func to get the sum over different thereads
        if (idx<st) s[idx] =s[idx]+s[st+idx];
        __syncthreads();
    }
    if (idx==0) *sum = s[0];
}

__global__ void softmax_kernel(float* output, int N,float* sum) {
    // Write code here
        int idx = blockIdx.x*blockDim.x +threadIdx.x;
    if (idx<N) {
        output[idx]/=*sum;
    }
    
}

extern "C" void solve(const float* input, float* output, int N) {
    float *d_max, *d_sum;
    cudaMalloc(&d_max,sizeof(float));
    cudaMalloc(&d_sum,sizeof(float));
    max_val<<<1,256>>>(input,d_max,N);
    sum_exp<<<1,256>>>(input,output,N,d_max,d_sum);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>( output, N,d_sum);
    cudaDeviceSynchronize();
}