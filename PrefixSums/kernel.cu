#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <stdio.h>



__global__ void prefixSum(int* input, int* output, int n) {
    extern __shared__ int temp[];  // Shared memory for intermediate results
    int tid = threadIdx.x;
    int pout = 0, pin = 1;  // Ping-pong buffers for in-place scan
    temp[tid] = (tid > 0) ? input[tid - 1] : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout; // Swap double buffer indices
        pin = 1 - pout;
        if (tid >= offset) {
            temp[pout * n + tid] = temp[pin * n + tid - offset] + temp[pin * n + tid];
        }
        else {
            temp[pout * n + tid] = temp[pin * n + tid];
        }
        __syncthreads();
    }
    output[tid] = temp[pout * n + tid];
}

int main() {
    int n = 8;
    int data[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    int* d_data;
    int* d_result;
    int* result = (int*)malloc(n * sizeof(int));

    cudaMalloc((void**)&d_data, n * sizeof(int));
    cudaMalloc((void**)&d_result, n * sizeof(int));

    cudaMemcpy(d_data, data, n * sizeof(int), cudaMemcpyHostToDevice);

    prefixSum <<<1, n, n * sizeof(int) >>> (d_data, d_result, n);

    cudaMemcpy(result, d_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%d ", result[i]);
    }
    printf("\n");

    cudaFree(d_data);
    cudaFree(d_result);

    return 0;
}

