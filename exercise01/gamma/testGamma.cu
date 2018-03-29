// ==========================================================================
// $Id$
// ==========================================================================
// (C)opyright: 2009-2010
//
//   Ulm University
// 
// Creator: Hendrik Lensch
// Email:   {hendrik.lensch,johannes.hanika}@uni-ulm.de
// ==========================================================================
// $Log$
// ==========================================================================

#include <stdio.h>
#include <vector_types.h>

#include "PPM.hh"

using namespace std;
using namespace ppm;

#define MAX_THREADS 128 

//-------------------------------------------------------------------------------

// specify the gamma value to be applied
__device__ __constant__ float gpuGamma[1];

__device__ float applyGamma(const float& _src, const float _gamma)
{
	return 255.0f * __powf(_src / 255.0f, _gamma);
}


__device__ float applydiff(const float& _src1, const float& _src2)
{
	return (fabsf(_src1 - _src2)) ;
}

__global__ void gammaKernel(float *_dst, const float* _src, int _w)
{
	int x = blockIdx.x * MAX_THREADS + threadIdx.x;
	int y = blockIdx.y;
	int pos = y * _w + x;

	if (x < _w)
	{
		_dst[pos] = applyGamma(_src[pos], gpuGamma[0]);
	}
}

__global__ void diffKernel(float *_dst, const float* _src1,const float* _src2, int _w)
{
	int x = blockIdx.x * MAX_THREADS + threadIdx.x;
	int y = blockIdx.y;
	int pos = y * _w + x;

	if (x < _w)
	{
		_dst[pos] = applydiff(_src1[pos],_src2[pos]);
	}
}

//-------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	int acount = 1; // parse command line

	if (argc < 5)
	{
		printf("usage: %s <inImg> <inImg2> <mode> <outImg>\n", argv[0]);
		exit(1);
	}

	float* img1;
	float* img2;
	

	int w, h;
	readPPM(argv[acount++], w, h, &img1);
	readPPM(argv[acount++], w, h, &img2);

	// float gamma = atof(argv[acount++]);
	// flag indicating weather the CPU or the GPU version should be executed
	bool gpuVersion = atoi(argv[acount++]);

	int nPix = w * h;

	float* gpuImg1;
	float* gpuImg2;

	float* gpuResImg;

	//-------------------------------------------------------------------------------
	printf("Executing the GPU Version\n");
	// copy the image to the device
	cudaMalloc((void**) &gpuImg1, nPix * 3 * sizeof(float));
	cudaMalloc((void**) &gpuImg2, nPix * 3 * sizeof(float));
	cudaMalloc((void**) &gpuResImg, nPix * 3 * sizeof(float));
	cudaMemcpy(gpuImg1, img1, nPix * 3 * sizeof(float),
			cudaMemcpyHostToDevice);
	cudaMemcpy(gpuImg2, img2, nPix * 3 * sizeof(float),
			cudaMemcpyHostToDevice);

	// copy gamma value to constant device memory
	// cudaMemcpyToSymbol(gpuGamma, &gamma, sizeof(float));

	// calculate the block dimensions
	dim3 threadBlock(MAX_THREADS);
	// select the number of blocks vertically (*3 because of RGB)
	dim3 blockGrid((w * 3) / MAX_THREADS + 1, h, 1);
	printf("bl/thr: %d  %d %d\n", blockGrid.x, blockGrid.y, threadBlock.x);

	//gammaKernel<<< blockGrid, threadBlock >>>(gpuResImg, gpuImg, w * 3);
	diffKernel<<< blockGrid, threadBlock >>>(gpuResImg, gpuImg1,gpuImg2, w * 3);
	cudaThreadSynchronize();

	// download result
	cudaMemcpy(img1, gpuResImg, nPix * 3 * sizeof(float),
			cudaMemcpyDeviceToHost);

	cudaFree(gpuResImg);
	cudaFree(gpuImg1);
	cudaFree(gpuImg2);

	writePPM(argv[acount++], w, h, (float*) img1);

	delete[] img1;
	delete[] img2;
	

	printf("  done\n");
}

