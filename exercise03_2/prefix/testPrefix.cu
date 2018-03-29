// ==========================================================================
// $Id$
// ==========================================================================
// (C)opyright: 2009
//
//   Ulm University
// 
// Creator: Hendrik Lensch, Holger Dammertz
// Email:   hendrik.lensch@uni-ulm.de, holger.dammertz@uni-ulm.de
// ==========================================================================
// $Log$
// ==========================================================================

#include <stdio.h>
#include <vector_types.h>
#include <vector>
#include <iostream>
#include <string>

#include "PPM.hh"

using namespace std;
using namespace ppm;
__device__ __constant__ float3 gpuClusterCol[2048];

#define THREADS 256
#define LOG_IMG_SIZE 8
#define IMG_SIZE 256
#define WINDOW 6


#define NUM_BANKS 16 
#define LOG_NUM_BANKS 4 
#ifdef ZERO_BANK_CONFLICTS 
#define CONFLICT_FREE_OFFSET(n) \ ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 
#else 
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS) 
#endif



void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}

// global texture reference 
texture<float4, 2, cudaReadModeElementType> texImg;

/*
__global__ void prefixsum(int *g_odata, int *g_idata, int n) 
{
	extern __shared__ float temp[];
	int thid = threadIdx.x;
	int offset=1;

	//temp[2*thid] = g_idata[2*thid];
	//temp[2*thid + 1] = g_idata[2*thid];

	int ai = thid; 
	int bi = thid + (n/2); 
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai); 
	int bankOffsetB = CONFLICT_FREE_OFFSET(ai); 
	temp[ai + bankOffsetA] = g_idata[ai];  
	temp[bi + bankOffsetB] = g_idata[bi];  

	for(int d = n>>1; d > 0; d >>= 1) 
	// build sum in place up the tree 
    	{ 
        	__syncthreads(); 
		if (thid < d)    
        	{ 
		int ai = offset*(2*thid+1)-1; 
		int bi = offset*(2*thid+2)-1; 
		ai += CONFLICT_FREE_OFFSET(ai); 
		bi += CONFLICT_FREE_OFFSET(bi); 
            	temp[bi] += temp[ai];         
  		} 
        	offset *= 2; 
    	} 

	if (thid == 0) { temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; } 
	//if (thid==0){ temp[n – 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; }

	for (int d = 1; d < n; d *= 2) 
	{
		offset >>= 1; 
		__syncthreads(); 
		if(thid < d) 
		{ 
			int ai = offset*(2*thid+1)-1; 
			int bi = offset*(2*thid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai); 
			bi += CONFLICT_FREE_OFFSET(bi); 
			float t   = temp[ai]; 
			temp[ai]  = temp[bi]; 
			temp[bi] += t; 
		}
	}

	__syncthreads();
	g_odata[ai] = temp[ai + bankOffsetA];  
	g_odata[bi] = temp[bi + bankOffsetB]; 
	//g_odata[2*thid] = temp[2*thid];
	//g_odata[2*thid+1] = temp[2*thid+1];

}

*/



/* The function measures for every pixel the distance to all
 clusters, and determines the clusterID of the nearest cluster
 center. It then colors the pixel in the cluster's color.

 The cluster centers are given as an array of linear indices into
 the vector image, i.e.    _clusterInfo[0] = (x_0 + y_0 * _w).

 */__global__ void voronoiKernel(float3 *_dst, int _w, int _h, int _nClusters,
		const int* _clusterInfo)
{
	// get the shared memory
	extern __shared__ int shm[];

	int nIter = _nClusters / THREADS + 1;
	// load cluster data
	for (int i = 0; i < nIter; ++i)
	{
		int pos = i * THREADS + threadIdx.x;
		if (pos < _nClusters)
		{
			shm[pos] = _clusterInfo[pos];
		}
	}

	__syncthreads();

	// compute the position within the image
	float x = blockIdx.x * blockDim.x + threadIdx.x;
	float y = blockIdx.y;

	int pos = x + y * _w;

	// determine which is the closest cluster
	float minDist = 1000000.;
	int minIdx = 0;
	for (int i = 0; i < _nClusters; ++i)
	{

		float yy = shm[i] >> LOG_IMG_SIZE;
		float xx = shm[i] % IMG_SIZE;

		float dist = (x - xx) * (x - xx) + (y - yy) * (y - yy);
		if (dist < minDist)
		{
			minDist = dist;
			minIdx = i;
		}
	}

	_dst[pos].x = gpuClusterCol[minIdx].x;
	_dst[pos].y = gpuClusterCol[minIdx].y;
	_dst[pos].z = gpuClusterCol[minIdx].z;

	// mark the center of each cluster
	if (minDist <= 2.)
	{
		_dst[pos].x = 255;
		_dst[pos].y = 0.;
		_dst[pos].z = 0.;
	}
}

__device__ float luminance(const float4& _col)
{
	return 0.299 * _col.x + 0.587 * _col.y + 0.114 * _col.z;
}

/** stores a 1 in _dst if the pixel's luminance is a maximum in the
 WINDOW x WINDOW neighborhood
 */__global__ void featureKernel(int *_dst, int _w, int _h)
{
	// compute the position within the image
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y;

	float lum = luminance(tex2D(texImg, x, y));

	bool maximum = false;

	if (lum > 20)
	{
		maximum = true;
		for (int v = y - WINDOW; v < y + WINDOW; ++v)
		{
			for (int u = x - WINDOW; u < x + WINDOW; ++u)
			{

				if (lum < luminance(tex2D(texImg, u, v)))
				{
					maximum = false;
				}

			}
		}
	}

	if (maximum)
	{
		_dst[x + y * _w] = 1;
	}
	else
	{
		_dst[x + y * _w] = 0;
	}
}


// !!! missing !!!
// Kernels for Prefix Sum calculation (compaction, spreading, possibly shifting)
// and for generating the gpuFeatureList from the prefix sum.

/* This program detects the local maxima in an image, writes their
 location into a vector and then computes the Voronoi diagram of the
 image given the detected local maxima as cluster centers.

 A Voronoi diagram simply colors every pixel with the color of the
 nearest cluster center. */


__global__ void prefixsum_down(int *data, int n,int offset,int stride)
{
	extern __shared__ float temp[];
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < n) {
		int thid = threadIdx.x;
		//int offset=0;
		//int stride=1;
		int incr = 1;

		temp[thid] = data[(x*stride) + offset]; //copy to each extern shared memory

		//temp[2*thid] = g_idata[2*thid];
		//temp[2*thid + 1] = g_idata[2*thid];
		/*

	int ai = thid;
	int bi = thid + (n/2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(ai);
	temp[ai + bankOffsetA] = g_idata[ai];
	temp[bi + bankOffsetB] = g_idata[bi];
		 */
		n = blockDim.x;

		for(int d = n>>1; d > 0; d >>= 1)
			// build sum in place up the tree
		{
			__syncthreads();
			if (thid < d)
			{
				int ai = incr*(2*thid+1)-1;
				int bi = incr + ai;
				//ai += CONFLICT_FREE_OFFSET(ai);
				//bi += CONFLICT_FREE_OFFSET(bi);
				temp[bi] += temp[ai];
			}
			incr *= 2;
		} //down summation done for one
		__syncthreads();

		data[(x*stride) + offset] = temp[thid]; // copied
	}
}






	//if (thid == 0) { temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; }
	//if (thid==0){ temp[n – 1] = 0; }
__global__ void prefixsum_up(int *data, int n,int offset,int stride)
{
	extern __shared__ float temp[];
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < n) {
		int thid = threadIdx.x;
		temp[thid] = data[(x*stride) + offset];

		int incr = blockDim.x;
		for (int d = 1; d < blockDim.x; d *= 2)
		{
			incr >>= 1;
			__syncthreads();
			if(thid < d)
			{
				int ai = 2*thid*incr;
				int bi = ai + incr;
				//ai += CONFLICT_FREE_OFFSET(ai);
				//bi += CONFLICT_FREE_OFFSET(bi);
				temp[bi] += temp[ai];
			}
		}

		__syncthreads();
		//g_odata[ai] = temp[ai + bankOffsetA];
		//g_odata[bi] = temp[bi + bankOffsetB];
		data[(x*stride) + offset] = temp[thid];

	}
}

__global__ void indicefinder(int *g_odata, int *g_idata, int n)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	//int thid = threadIdx.x;
	//temp[thid] = g_idata[x]*stride + offset;


	if (g_idata[x]!=g_idata[x+1])
	{

		g_odata[g_idata[x]]=x;

	}



}



__global__ void prefixsum_placearray(int *g_odata, int *i_data, int *old)
{
	extern __shared__ float temp[];
	int offset = THREADS -1;
	int stride = THREADS;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int thid = threadIdx.x;
	temp[thid] = i_data[thid];

	g_odata[x]=old[x];
	g_odata[x*stride + offset] = temp[thid];


	}
__global__ void shiftbyzero(int *g_odata, int *i_data)
{

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	g_odata[x+1]=i_data[x];
	if (x == 0)
		g_odata[0] = 0;
	//g_odata[x]*stride + offset = temp[thid];
	}



int main(int argc, char* argv[])
{

	// parse command line
	int acount = 1;
	if (argc < 3)
	{
		printf("usage: testPrefix <inImg> <outImg> <mode>");
		exit(1);
	}
	string inName(argv[acount++]);
	string outName(argv[acount++]);
	int mode = atoi(argv[acount++]);

	// Load the input image
	float* cpuImage;
	int w, h;
	readPPM(inName.c_str(), w, h, &cpuImage);
	int nPix = w * h;
//	int B =512; //number of elements processed in a block
//	int blocks = nPix/B;
//	int threads = B/2;

	// Allocate GPU memory
	int* gpuFeatureImg; // Contains 1 for a feature, 0 else
						// Can be used to do the reduction step of prefix sum calculation in place
	int* gpuPrefixSumShifted; // Output buffer containing the prefix sum
							  // Shifted by 1 since it contains 0 as first element by definition
	int* gpuFeatureList; // List of pixel indices where features can be found.



	int* preshiftgpu;
//	int* gpuPrefixSumDown_small;
//	int* gpuPrefixSumDown;
	int* gpuPrefixSumUp_small;
	int* gpuPrefixSumUp;
//	int* finallist;
	int* gpuPrefixSumShifted2;
	cudaMalloc((void**) &preshiftgpu, (nPix) * sizeof(int));
	cudaMalloc((void**) &gpuPrefixSumUp_small, (THREADS) * sizeof(int));
	cudaMalloc((void**) &gpuPrefixSumUp, (nPix + 1) * sizeof(int));
	cudaMalloc((void**) &gpuPrefixSumShifted2, (nPix + 1) * sizeof(int));
	//cudaMalloc((void**) &finallist, (THREADS) * sizeof(int));




	float3* gpuVoronoiImg; // Final rgb output image
	cudaMalloc((void**) &gpuFeatureImg, (nPix) * sizeof(int));

	cudaMalloc((void**) &gpuPrefixSumShifted, (nPix + 1) * sizeof(int));
	//cudaMalloc((void**) &gpuPrefixSumShifted, (nPix) * sizeof(int));
	cudaMalloc((void**) &gpuFeatureList, 10000 * sizeof(int));

	cudaMalloc((void**) &gpuVoronoiImg, nPix * 3 * sizeof(float));

	// color map for the cluster
	float clusterCol[2048 * 3];
	float* ci = clusterCol;
	for (int i = 0; i < 2048; ++i, ci += 3)
	{
		ci[0] = 32 * i % 256;
		ci[1] = (10 * i + 128) % 256;
		ci[2] = (40 * i + 255) % 256;
	}

	cudaMemcpyToSymbol(gpuClusterCol, clusterCol, 2048 * 3 * sizeof(float));

	cudaArray* gpuTex;
	cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float4>();
	cudaMallocArray(&gpuTex, &floatTex, w, h);

	// pad to float4 for faster access
	float* img4 = new float[w * h * 4];

	for (int i = 0; i < w * h; ++i)
	{
		img4[4 * i] = cpuImage[3 * i];
		img4[4 * i + 1] = cpuImage[3 * i + 1];
		img4[4 * i + 2] = cpuImage[3 * i + 2];
		img4[4 * i + 3] = 0.;
	}

	// upload to array

	cudaMemcpyToArray(gpuTex, 0, 0, img4, w * h * 4 * sizeof(float),
			cudaMemcpyHostToDevice);

	// bind as texture
	cudaBindTextureToArray(texImg, gpuTex, floatTex);

	cout << "setup texture" << endl;
	cout.flush();

	// calculate the block dimensions
	dim3 threadBlock(THREADS);
	dim3 numBlock(w*h / THREADS);
	dim3 numBlock2(((w*h) + 1) / THREADS);
	dim3 blockGrid(w / THREADS, h, 1);

	printf("blockDim: %d  %d \n", threadBlock.x, threadBlock.y);
	printf("gridDim: %d  %d \n", blockGrid.x, blockGrid.y);

	featureKernel<<<blockGrid, threadBlock>>>(gpuFeatureImg, w, h);
	// gpuFeature

	// variable to store the number of detected features = the number of clusters
	int nFeatures;

	if (mode == 0)
	{
		////////////////////////////////////////////////////////////
		// CPU compaction:
		////////////////////////////////////////////////////////////

		// download result

		cudaMemcpy(cpuImage, gpuFeatureImg, nPix * sizeof(float),
				cudaMemcpyDeviceToHost);

		std::vector<int> features;

		float* ii = cpuImage;
		for (int i = 0; i < nPix; ++i, ++ii)
		{
			if (*ii > 0)
			{
				features.push_back(i);
			}
		}

		cout << "nFeatures: " << features.size() << endl;

		nFeatures = features.size();
		// upload feature vector

		cudaMemcpy(gpuFeatureList, &(features[0]), nFeatures * sizeof(int),
				cudaMemcpyHostToDevice);
	}
	else
	{

		int offset=0;
		int stride = 1;

		int test_size =  nPix;

		int * test_input = new int[test_size+1];

		prefixsum_down<<<numBlock, threadBlock, THREADS * sizeof(int)>>>(gpuFeatureImg, nPix, offset, stride);
		cudaThreadSynchronize();

		offset = THREADS -1;
		stride = THREADS;

		prefixsum_down<<<1, threadBlock, THREADS * sizeof(int)>>>(gpuFeatureImg, nPix, offset, stride); //initialize sumdown, 1 or many?
		cudaThreadSynchronize();



		shiftbyzero<<<numBlock,threadBlock>>>(gpuPrefixSumShifted, gpuFeatureImg); //shifted, ready to be put into down sweep
		cudaThreadSynchronize();
		offset =0;
		stride =THREADS;


		prefixsum_up<<<1, threadBlock, THREADS * sizeof(int)>>>(gpuPrefixSumShifted, nPix + 1,offset, stride);
		cudaThreadSynchronize();
		offset =0;
		stride =1;

		prefixsum_up<<<numBlock2, threadBlock, THREADS * sizeof(int)>>>(gpuPrefixSumShifted, nPix + 1, offset, stride); // preferably prefixsum obtained
		cudaThreadSynchronize();



		cudaMemcpy(&nFeatures, &gpuPrefixSumShifted[nPix], sizeof(int), cudaMemcpyDeviceToHost);
		printf("this stage cleared \n");
		printf("features found in GPU: %d \n", nFeatures);
		printf("Should be: %d \n", nPix);


		//now find the indices
//		cudaMalloc((void**) &finallist, (nFeatures) * sizeof(int));
		indicefinder<<<numBlock,threadBlock>>>(gpuFeatureList, gpuPrefixSumShifted, nPix+1);
		cudaThreadSynchronize();
		checkCUDAError("Kernel is Broken");
		for (int i = 0; i < test_size; i++)
			test_input[i] = 0;
		cudaMemcpy(test_input, gpuFeatureList, nFeatures*sizeof(int), cudaMemcpyDeviceToHost);

	
	}

	// now compute the Voronoi Diagram around the detected features.
	voronoiKernel<<<blockGrid, threadBlock, nFeatures * sizeof(int)>>>(
			gpuVoronoiImg, w, h, nFeatures, gpuFeatureList);
	

	// download final voronoi image.

	cudaMemcpy(cpuImage, gpuVoronoiImg, nPix * 3 * sizeof(float),
			cudaMemcpyDeviceToHost);
	// Write to disk
	writePPM(outName.c_str(), w, h, (float*) cpuImage);

	// Cleanup
	cudaUnbindTexture(texImg);
	cudaFreeArray(gpuTex);
	cudaFree(gpuFeatureList);
	cudaFree(gpuFeatureImg);
	cudaFree(gpuPrefixSumShifted);
	cudaFree(gpuVoronoiImg);

	delete[] cpuImage;
	delete[] img4;

	printf("done\n");

}
