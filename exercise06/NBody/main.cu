#include "gltools.h"
#include "Tools.h"

#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>

using namespace std;

#define GUI
#define NUM_FRAMES 250

#define THREADS_PER_BLOCK 128
#define EPS_2 0.00001f
#define GRAVITY 0.00000001f

float randF(const float min = 0.0f, const float max = 1.0f)
{
	int randI = rand();
	float randF = (float) randI / (float) RAND_MAX;
	float result = min + randF * (max - min);

	return result;
}

inline __device__ float2 operator+(const float2 op1, const float2 op2)
{
	return make_float2(op1.x + op2.x, op1.y + op2.y);
}

inline __device__ float2 operator-(const float2 op1, const float2 op2)
{
	return make_float2(op1.x - op2.x, op1.y - op2.y);
}

inline __device__ float2 operator*(const float2 op1, const float op2)
{
	return make_float2(op1.x * op2, op1.y * op2);
}

inline __device__ float2 operator/(const float2 op1, const float op2)
{
	return make_float2(op1.x / op2, op1.y / op2);
}

inline __device__ void operator+=(float2 &a, const float2 b)
{
	a.x += b.x;
	a.y += b.y;
}


__global__ void acceleration(float2 *acc, float2 *pos, float *mass, int n)
{
	//extern __shared__ float temp[];

	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x<n)
	{
		float2 myPos = pos[x];
		float2 sum;
		sum.x = 0;
		sum.y = 0;
		for (int i=0;i<n;i++) {
			float2 position = pos[i];

			float2 dis = (position-myPos);
			float l2_norm = sqrtf(powf((position.x-myPos.x),2) + powf((position.y-myPos.y),2));
			sum += (dis * mass[i])/powf(((l2_norm * l2_norm) + EPS_2),1.5);
			//sum.y += (mass[i] * dis.y)/powf(((l2_norm * l2_norm) + EPS_2),1.5);
		}
		sum = sum * GRAVITY;
		acc[x].x = sum.x;
		acc[x].y = sum.y;
	}
}

__global__ void velocity(float2 *vel, float2 *acc,int n)
{
	//extern __shared__ float temp[];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x<n){
		vel[x]+=acc[x];}
}

__global__ void positionKern(float2 *pos, float2 *vel,int n)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x<n){
		pos[x]+=vel[x];}
}



int main(int argc, char **argv)
{
	if (argc != 2)
	{
		cout << "Usage: " << argv[0] << " <numBodies>" << endl;
		return 1;
	}
	unsigned int numBodies = atoi(argv[1]);
	unsigned int numBlocks = numBodies / THREADS_PER_BLOCK;
	numBodies = numBlocks * THREADS_PER_BLOCK;

	// allocate memory
	float2* hPositions = new float2[numBodies];
	float2* hVelocities = new float2[numBodies];
	float* hMasses = new float[numBodies];

	// Initialize Positions and speed
	for (unsigned int i = 0; i < numBodies; i++)
	{
		hPositions[i].x = randF(-1.0, 1.0);
		hPositions[i].y = randF(-1.0, 1.0);
		hVelocities[i].x = hPositions[i].y * 0.007f + randF(0.001f, -0.001f);
		hVelocities[i].y = -hPositions[i].x * 0.007f + randF(0.001f, -0.001f);
		hMasses[i] = randF(0.0f, 1.0f) * 10000.0f / (float) numBodies;
	}

	// float
	float2* gPositions;
	float2* gVelocities;
	float* gMasses;
	float2* gAcc;
	cudaMalloc((void**) &gPositions, (numBodies) * sizeof(float2));
	cudaMalloc((void**) &gVelocities, (numBodies) * sizeof(float2));
	cudaMalloc((void**) &gMasses, (numBodies) * sizeof(float));
	cudaMalloc((void**) &gAcc, (numBodies) * sizeof(float2));

	cudaMemcpy(gPositions,hPositions,numBodies*sizeof(float2),cudaMemcpyHostToDevice);
	cudaMemcpy(gVelocities,hVelocities,numBodies*sizeof(float2),cudaMemcpyHostToDevice);
	cudaMemcpy(gMasses,hMasses,numBodies*sizeof(float),cudaMemcpyHostToDevice);


	//cudaArray* gPositions;
	//cudaChannelFormatDesc floatTex1 = cudaCreateChannelDesc<float2>();
	//cudaMallocArray(&gPositions, &floatTex1, numBodies);

	//cudaArray* gMasses;
	//cudaChannelFormatDesc floatTex2 = cudaCreateChannelDesc<float>();
	//cudaMallocArray(&gMasses, &floatTex2, numBodies);

	//cudaMemcpyToArray(gPositions, 0, 0, hPositions, numBodies * sizeof(float2),
	//			cudaMemcpyHostToDevice);
	//cudaMemcpyToArray(gMasses, 0, 0, hMasses, numBodies * sizeof(float),
	//				cudaMemcpyHostToDevice);



	// TODO 1: Allocate GPU memory for
	// - Positions,
	// - Velocities,
	// - Accelerations and
	// - Masses
	// of all bodies and initialize them from the CPU arrays (where available).

	// Free host memory not needed again

	delete[] hMasses;

	// Initialize OpenGL rendering
#ifdef GUI
	initGL();
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	GLuint sp = createShaderProgram("white.vs", 0, 0, 0, "white.fs");

	GLuint vb;
	glGenBuffers(1, &vb);
	GL_CHECK_ERROR;
	glBindBuffer(GL_ARRAY_BUFFER, vb);
	GL_CHECK_ERROR;
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * numBodies, hPositions,
			GL_STATIC_DRAW);
	GL_CHECK_ERROR;
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	GL_CHECK_ERROR;

	GLuint va;
	glGenVertexArrays(1, &va);
	GL_CHECK_ERROR;
	glBindVertexArray(va);
	GL_CHECK_ERROR;
	glBindBuffer(GL_ARRAY_BUFFER, vb);
	GL_CHECK_ERROR;
	glEnableVertexAttribArray(glGetAttribLocation(sp, "inPosition"));
	GL_CHECK_ERROR;
	glVertexAttribPointer(glGetAttribLocation(sp, "inPosition"), 2, GL_FLOAT,
			GL_FALSE, 0, 0);
	GL_CHECK_ERROR;
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	GL_CHECK_ERROR;
	glBindVertexArray(0);
	GL_CHECK_ERROR;
#endif

	// Calculate
	for(unsigned int t = 0; t < NUM_FRAMES; t++)
	{
		__int64_t computeStart = continuousTimeNs();

		// TODO 3: Update accelerations of all bodies here.
		acceleration<<<numBlocks, THREADS_PER_BLOCK>>>(gAcc, gPositions,gMasses,numBodies);
		cudaThreadSynchronize();
		//cudaMemcpy(hVelocity,gVelocity,numBodies*sizeof(float2),cudaDeviceToHost);

		velocity<<<numBlocks, THREADS_PER_BLOCK>>>(gVelocities, gAcc,numBodies);
		cudaThreadSynchronize();
		positionKern<<<numBlocks, THREADS_PER_BLOCK>>>(gPositions, gVelocities,numBodies);
		cudaThreadSynchronize();
		cudaMemcpy(hPositions,gPositions,numBodies*sizeof(float2),cudaMemcpyDeviceToHost);

		// TODO 4: Update velocities and positions of all bodies here.


		cudaThreadSynchronize();
		cout << "Frame compute time: " << (continuousTimeNs() - computeStart)
				<< "ns" << endl;

		// TODO 5: Download the updated positions into the hPositions array for rendering.

#ifdef GUI
		// Upload positions to OpenGL
		glBindBuffer(GL_ARRAY_BUFFER, vb);
		GL_CHECK_ERROR;
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * numBodies, hPositions,
				GL_STATIC_DRAW);
		GL_CHECK_ERROR;
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		GL_CHECK_ERROR;

		// Draw
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		GL_CHECK_ERROR;
		glUseProgram(sp);
		GL_CHECK_ERROR;
		glBindVertexArray(va);
		GL_CHECK_ERROR;
		glDrawArrays(GL_POINTS, 0, numBodies);
		GL_CHECK_ERROR;
		glBindVertexArray(0);
		GL_CHECK_ERROR;
		glUseProgram(0);
		GL_CHECK_ERROR;
		swapBuffers();
#endif
	}

#ifdef GUI
	cout << "Done." << endl;
	sleep(2);
#endif

	// Clean up
#ifdef GUI
	glDeleteProgram(sp);
	GL_CHECK_ERROR;
	glDeleteVertexArrays(1, &va);
	GL_CHECK_ERROR;
	glDeleteBuffers(1, &vb);
	GL_CHECK_ERROR;

	glDeleteProgram(sp);
	exitGL();
#endif

	// TODO 2: Clean up your allocated memory

	delete[] hPositions;
	delete[] hVelocities;
	cudaFree(gAcc);
	cudaFree(gPositions);
	cudaFree(gVelocities);
	cudaFree(gMasses);

}

