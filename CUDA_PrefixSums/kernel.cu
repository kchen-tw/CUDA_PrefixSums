#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// CUDA 核心函數，使用 shared memory 進行前綴和計算
__global__  void prefixSum_divide(int* input, int* output) {
	extern __shared__ int temp[];  // Shared memory for intermediate results
	int n = blockDim.x;
	int tid = threadIdx.x;
	int sid = blockIdx.x * n + tid;
	int pout = 0, pin = 1;  // Ping-pong buffers for in-place scan
	temp[tid] = input[sid];
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
	output[sid] = temp[pout * n + tid];
}

__global__ void add_lastValue(int* output, int last_value_idx) {
	int tid = threadIdx.x;
	int sid = blockIdx.x * blockDim.x + tid;
	output[last_value_idx + 1 + sid] += output[last_value_idx];
}

__global__  void prefixSum_conquer(int* output, int n, int numBlocks) {
	
	// 合併的次數
	int numConquer = log2((double)numBlocks);
	//numConquer = 5;
	for (int i = 1; i <= numConquer; i++) {
		// 每次合併的 size 會是上次的兩倍
		int mergeSize = 1 << (i - 1);

		// j 為每次合併第奇數個 blockid
		// i=1, j=1,3,5,7,9,...
		// i=2, j=2,6,10,14,...
		// i=3, j=4,12,20,28,...
		for (int j = mergeSize; j < numBlocks; j += 2 * mergeSize) {
			
			// 奇數block最後一個數值的 index
			int last_value_idx = j * n - 1;

			// 將奇數block的最後一個數值與後面 mergeSize*n 個數值相加
			add_lastValue << <mergeSize, n >> > (output, last_value_idx);
		}
		__syncthreads();
	}
}

__global__ void prefixSum_kernal(int* input, int* output, int N, int maxThreadsPerBlock) {
	int threadsPerBlock = (N < maxThreadsPerBlock) ? N : maxThreadsPerBlock;
	int numBlocks = (N + threadsPerBlock -1) / threadsPerBlock;

	prefixSum_divide << <numBlocks, threadsPerBlock, threadsPerBlock * 2 * sizeof(int) >> > (input, output);
	prefixSum_conquer << <1, 1 >> > (output, threadsPerBlock, numBlocks);
}

// 產生亂數陣列
void generateRandom(int array[], int size, int min, int max) {
	random_device rd;  // 隨機數種子
	mt19937 gen(rd()); // 使用 Mersenne Twister 算法
	uniform_int_distribution<> dis(min, max); // 定義均勻分佈

	for (int i = 0; i < size; ++i) {
		array[i] = dis(gen); // 產生隨機數並寫入陣列中
	}
}

// CPU 的 prefixSum
void prefixSum_cpu(int* input, int* output, int n) {
	if (n <= 0) {
		// 空陣列或無效長度，不進行計算
		return;
	}

	output[0] = input[0];  // 第一個元素的前綴和就是它本身

	for (int i = 1; i < n; ++i) {
		output[i] = output[i - 1] + input[i];  // 計算前綴和
	}
}

// 比較兩個陣列是否相同0
bool areArraysEqual(int* arr1, int* arr2, int n) {
	for (int i = 0; i < n; ++i) {
		if (arr1[i] != arr2[i]) {
			return false;  // 一旦發現不同，就返回 false
		}
	}
	return true;  // 如果全部元素都相同，返回 true
}

int main() {

	const int N = 1 << 14; // 根據您的需求調整大小

	// 在主機上分配記憶體
	int* h_input = new int[N];
	int* h_output = new int[N];
	int* h_output_cpu = new int[N];

	// 在設備上分配記憶體
	int* d_input, * d_output;
	cudaMalloc((void**)&d_input, N * sizeof(int));
	cudaMalloc((void**)&d_output, N * sizeof(int));


	// 初始化輸入陣列
	//for (int i = 0; i < N; ++i) {
	//	h_input[i] = 1; 
	//}

	// 產生亂數陣列
	generateRandom(h_input, N, 0, 100);

	// 將輸入數組複製到設備
	cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

	// 獲取當前 CUDA 設備的 ID
	int deviceId;
	cudaGetDevice(&deviceId);

	// 獲取硬體的靜態限制每個線程塊可容納的最大線程數
	int maxThreadsPerBlock;
	cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, deviceId);

	prefixSum_kernal << < 1, 1 >> > (d_input, d_input, N, maxThreadsPerBlock);

	// // 同步所有 CUDA kernel 的執行
	cudaDeviceSynchronize();

	// 從設備將結果複製回主機
	cudaMemcpy(h_output, d_input, N * sizeof(int), cudaMemcpyDeviceToHost);

	//用CPU計算 prefixSums
	prefixSum_cpu(h_input, h_output_cpu, N);

	// 比較cpu與gpu計算內容是否相同
	if (areArraysEqual(h_output, h_output_cpu, N)) {
	    cout << "CPU and GPU are equal.\n";
	}
	else {
	    cout << "CPU and GPU are not equal.\n";
	}

	// 將結果寫入檔案
	ofstream prefixsumsFile("prefix_sums.txt");
	for (int i = 0; i < N; ++i) {
		prefixsumsFile << h_output[i] << endl;
	}
	prefixsumsFile.close();
	cout << "the result of prefix sums 'prefix_sums.txt' created successfully." << endl;


	// 釋放記憶體
	delete[] h_input;
	delete[] h_output;
	delete[] h_output_cpu;

	cudaFree(d_input);
	cudaFree(d_output);

	return 0;
}
