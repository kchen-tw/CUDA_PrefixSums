#include <iostream>
#include <fstream>
#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// CUDA 核心函數，使用 shared memory 進行前綴和計算
__global__  void prefixSum_divide(int* input, int* output, int n) {
	extern __shared__ int temp[];  // Shared memory for intermediate results
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

__global__ void add_lastValue(int* output, int last_value_idx, int value_idx) {
	output[value_idx + threadIdx.x] += output[last_value_idx];
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

			// 因為每次合併會是上次的兩倍，但一次只能相加 n 筆資料，所以需要個迴圈 k 去相加後面 mergeSize 次
			for (int k = 0; k < mergeSize; k++) {
				//printf("i=%d, j=%d, ,k=%d, j*n-1=%d, (j+k)*n=%d~%d\n", i, j, k, j * n - 1, (j + k) * n, (j + k) * n + 1024);

				// 要相加的啟始index
				int value_idx = (j + k) * n; 

				// 執行 n 個threads, 同時計算last_value與後面 n 個值相加
				add_lastValue << <1, n >> > (output, last_value_idx, value_idx);
			}
			
		}
		__syncthreads();
	}
}

__global__ void prefixSum_kernal(int* input, int* output, int N) {
	int threadsPerBlock = (N < 1024) ? N : 1024;
	int numBlocks = (N % threadsPerBlock == 0) ? N / threadsPerBlock : N / threadsPerBlock + 1;
	prefixSum_divide << <numBlocks, threadsPerBlock, threadsPerBlock * 2 * sizeof(int) >> > (input, output, threadsPerBlock);
	prefixSum_conquer << <1, 1 >> > (output, threadsPerBlock, numBlocks);
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

	const int N = 1 << 25; // 根據您的需求調整大小

	// 在主機上分配記憶體
	int* h_input = new int[N];
	int* h_output = new int[N];
	int* h_output_cpu = new int[N];

	// 在設備上分配記憶體
	int* d_input, * d_output;
	cudaMalloc((void**)&d_input, N * sizeof(int));
	cudaMalloc((void**)&d_output, N * sizeof(int));


	// 初始化輸入數組
	for (int i = 0; i < N; ++i) {
		h_input[i] = 1; // 您可以使用您自己的值進行初始化
	}

	// 將輸入數組複製到設備
	cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

	//int threadSize = (N < 1024) ? N : 1024;

	// 設置網格維度
	//int blockSize = (N % threadSize == 0) ? N / threadSize : N / threadSize + 1;

	// 運行核心函數，使用 shared memory 進行前綴和計算
	//prefixSum << <gridSize, blockSize, N * sizeof(int) >> > (d_input, d_output, N);
	//prefixSum<<<blockSize, threadSize, threadSize * 2 * sizeof(int)>>>(d_input, d_input, threadSize);

	prefixSum_kernal << < 1, 1 >> > (d_input, d_input, N);

	// 等待GPU執行完畢
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
	//ofstream prefixsumsFile("prefix_sums.txt");
	//for (int i = 0; i < N; ++i) {
	//	prefixsumsFile << h_output[i] << endl;
	//}
	//prefixsumsFile.close();
	//cout << "the result of prefix sums 'prefix_sums.txt' created successfully." << endl;


	// 釋放記憶體
	delete[] h_input;
	delete[] h_output;
	delete[] h_output_cpu;

	cudaFree(d_input);
	cudaFree(d_output);

	return 0;
}
