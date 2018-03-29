#include "Tools.h"

#include <iostream>
#include <iomanip>

using namespace std;

//#define VERBOSE // Prints input matrix and results. Only uncomment for small matrix sizes!
#define RUN_CPU // Runs CPU code for reference (slow!!!)
#define N 1024 // Must be a multiple of THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 32 // per axis -> block has this value squared threads.
void multiplyMatrix(float* result, const float* a, const float* b, const int n)
{
	for (unsigned int i = 0; i < n; i++)
	{
		for (unsigned int j = 0; j < n; j++)
		{
			result[i * n + j] = 0.0f;
			for (unsigned int k = 0; k < n; k++)
			{
				result[i * n + j] += a[i * n + k] * b[k * n + j];
			}
		}
	}
}

void dumpMatrix(const float* m, const int n)
{
	for (unsigned int i = 0; i < n; i++)
	{
		for (unsigned int j = 0; j < n; j++)
		{
			cout << setw(3) << setprecision(3) << m[i * n + j] << " ";
		}
		cout << endl;
	}
}

float randF(const float min = 0.0f, const float max = 1.0f)
{
	int randI = rand();
	float randF = (float) randI / (float) RAND_MAX;
	float result = min + randF * (max - min);

	return result;
}

__global__ void multiplyMatrixGpu1(float* result, const float* a,
		const float* b, const int n)
{
		int row = blockIdx.y * blockDim.y + threadIdx.y;
	    int col = blockIdx.x * blockDim.x + threadIdx.x;
	    float sum = 0;
	    if( col < n && row < n)
	    {
	        for(int i = 0; i < n; i++)
	        {
	            sum += a[row * n + i] * b[i * n + col];
	        }
	        result[row * n + col] = sum;
	}
}

__global__ void multiplyMatrixGpu2(float* d_result, const float* d_a,
		const float* d_b, const int n)
{
		const int BLOCK_SIZE = THREADS_PER_BLOCK;
	 	__shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE];
	    __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE];

	    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	    float tmp = 0.0;
	    int idx;

	    for (int sub = 0; sub < gridDim.x; ++sub)
	    {
	        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
	        if(idx >= n*n)
	        {
	            // n may not divisible by BLOCK_SIZE
	            tile_a[threadIdx.y][threadIdx.x] = 0;
	        }
	        else
	        {
	            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
	        }

	        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
	        if(idx >= n*n)
	        {
	            tile_b[threadIdx.y][threadIdx.x] = 0;
	        }
	        else
	        {
	            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
	        }
	        __syncthreads();

	        for (int k = 0; k < BLOCK_SIZE; ++k)
	        {
	            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
	        }
	        __syncthreads();
	    }
	    if(row < n && col < n)
	    {
	        d_result[row * n + col] = tmp;
	}
}

int main(int argc, char **argv)
{
	__int64_t startTime;
	__int64_t endTime;

	// Allocate all memory
	float* hM1 = new float[N * N];
	float* hM2 = new float[N * N];
	float* hMR = new float[N * N];
	float* gM1;
	cudaMalloc(&gM1, sizeof(float) * N * N);
	float* gM2;
	cudaMalloc(&gM2, sizeof(float) * N * N);
	float* gMR;
	cudaMalloc(&gMR, sizeof(float) * N * N);

	// Initialize matrices and upload to CUDA
	for (unsigned int n = 0; n < N * N; n++)
	{
		hM1[n] = randF(-1.0, 1.0);
		hM2[n] = randF(-1.0, 1.0);
	}
	cudaMemcpy(gM1, hM1, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(gM2, hM2, sizeof(int) * N * N, cudaMemcpyHostToDevice);
#ifdef VERBOSE
	cout << "Input Matrices:" << endl;
	dumpMatrix(hM1, N);
	cout << endl;
	dumpMatrix(hM2, N);
	cout << endl << endl;
#endif

#ifdef RUN_CPU
	// Calculations on CPU
	startTime = continuousTimeNs();
	multiplyMatrix(hMR, hM1, hM2, N);
	endTime = continuousTimeNs();
#ifdef VERBOSE
	cout << "CPU:" << endl;
	dumpMatrix(hMR, N);
	cout << endl;
#endif
	cout << "CPU time: " << (endTime - startTime) << "ns" << endl;
#endif

	// Calculations on GPU
	int blocksPerGridX =
			N % THREADS_PER_BLOCK == 0 ?
					N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
	int blocksPerGridY =
			N % THREADS_PER_BLOCK == 0 ?
					N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
	startTime = continuousTimeNs();
	multiplyMatrixGpu1<<<dim3(blocksPerGridX, blocksPerGridY, 1),
			dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)>>>(gMR, gM1, gM2, N);
	cudaDeviceSynchronize();
	endTime = continuousTimeNs();
	cudaMemcpy(hMR, gMR, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
#ifdef VERBOSE
	cout << "GPU simple:" << endl;
	dumpMatrix(hMR, N);
	cout << endl;
#endif
	cout << "GPU simple time: " << (endTime - startTime) << "ns" << endl;
	startTime = continuousTimeNs();
	multiplyMatrixGpu2<<<dim3(blocksPerGridX, blocksPerGridY, 1),
			dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)>>>(gMR, gM1, gM2, N);
	cudaDeviceSynchronize();
	endTime = continuousTimeNs();
	cudaMemcpy(hMR, gMR, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
#ifdef VERBOSE
	cout << "GPU advanced:" << endl;
	dumpMatrix(hMR, N);
	cout << endl;
#endif
	cout << "GPU advanced time: " << (endTime - startTime) << "ns" << endl;

	// Free all memory
	cudaFree(gM1);
	cudaFree(gM2);
	cudaFree(gMR);
	delete[] hM1;
	delete[] hM2;
	delete[] hMR;
}
