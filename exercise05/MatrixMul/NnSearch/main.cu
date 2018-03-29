#include "Tools.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <math.h>

using namespace std;

#define BF // Don't use for benchmarking!
#define KD
#define KD_GPU

/// Numbers for testing
#define NUM_POINTS 20
#define NUM_QUERIES 10

/// Numbers for benchmarking
//#define NUM_POINTS 1000000
//#define NUM_QUERIES 100000

#define THREADS_PER_BLOCK 512

//#define VERBOSE

struct Point
{
	float x;
	float y;

	// TODO: Modify methods so that points can be used in kernels, too.

	float operator[](const unsigned int index) const
	{
		return *((float*) this + index);
	}

	float & operator[](const unsigned int index)
	{
		return *((float*) this + index);
	}

	Point operator-(const Point& other) const
	{
		Point result;
		result.x = x - other.x;
		result.y = y - other.y;

		return result;
	}

	float length() const
	{
		return sqrtf(x * x + y * y);
	}
};

std::ostream& operator<<(std::ostream& os, const Point& point)
{
	os << "(" << point.x << "," << point.y << ")";
	return os;
}

struct KdNode
{
	const static char X = 0;
	const static char Y = 1;
	const static char None = 2;

	int PointIdx;
	int RightChildIdx;
	char SplitDim;
};

const char KdNode::X;
const char KdNode::Y;

struct NodeComparator
{
	char SplitDim;
	const vector<Point> &Points;

	NodeComparator(const char splitDim, const vector<Point> &points) :
			SplitDim(splitDim), Points(points)
	{
	}

	bool operator()(const KdNode &i, const KdNode &j)
	{
		return (Points[i.PointIdx][SplitDim] < Points[j.PointIdx][SplitDim]);
	}
};

float randF(const float min = 0.0f, const float max = 1.0f)
{
	int randI = rand();
	float randF = (float) randI / (float) RAND_MAX;
	float result = min + randF * (max - min);

	return result;
}

void dumpPart(vector<KdNode> &nodes, const vector<Point> &points,
		const int fromNode, const int toNode, const char splitDim)
{
#ifdef VERBOSE
	cout << "Index " << fromNode << " to " << toNode << ": ";
	for (unsigned int n = fromNode; n < toNode; n++)
	{
		if (n > fromNode)
			cout << ", ";
		cout << points[nodes[n].PointIdx][splitDim];
	}
	cout << endl;
#endif
}

void makeKdTree(vector<KdNode> &nodes, const vector<Point> &points,
		const int fromNode, const int toNode, const char splitDim)
{
	// Check base case
	if (fromNode + 1 == toNode)
	{
		dumpPart(nodes, points, fromNode, toNode, splitDim);
		return;
	}

	// Sort all nodes currently processed
	sort(nodes.begin() + fromNode, nodes.begin() + toNode,
			NodeComparator(splitDim, points));

	// Check special case: just one child
	if (fromNode + 2 == toNode)
	{
		// Make the larger node the right child of the smaller node
		nodes[fromNode].RightChildIdx = fromNode + 1;
		nodes[fromNode].SplitDim = splitDim;

		// Done: Right child is correctly set to leaf because all nodes are initialized as leafs.
		dumpPart(nodes, points, fromNode, toNode, splitDim);
		return;
	}

	// Recursive case: At least three nodes -> one central one which has at least two children

	// Find the index where the nodes are split
	int splitNode = (fromNode + toNode) / 2;

	// Pull the split node to the front
	int tmp = nodes[fromNode].PointIdx;
	nodes[fromNode].PointIdx = nodes[splitNode].PointIdx;
	nodes[splitNode].PointIdx = tmp;

	// Setup the split node
	nodes[fromNode].RightChildIdx = splitNode + 1;
	nodes[fromNode].SplitDim = splitDim;

	dumpPart(nodes, points, fromNode, toNode, splitDim);

	// Recursively process left and right part
	char newSplitDim = (splitDim == KdNode::X) ? KdNode::Y : KdNode::X;
	makeKdTree(nodes, points, fromNode + 1, splitNode + 1, newSplitDim);
	makeKdTree(nodes, points, splitNode + 1, toNode, newSplitDim);
}

vector<KdNode> makeKdTree(const vector<Point> &points)
{
	// Create a node for every point
	vector<KdNode> nodes(points.size());
	for (unsigned int n = 0; n < nodes.size(); n++)
	{
		nodes[n].PointIdx = n;
		nodes[n].RightChildIdx = -1;
		nodes[n].SplitDim = KdNode::None;
	}

	makeKdTree(nodes, points, 0, nodes.size(), KdNode::X);

	return nodes;
}

unsigned int findNnBruteForce(const vector<Point> &points, const Point& query)
{
	unsigned int bestIdx = 0;
	for (unsigned int p = 1; p < NUM_POINTS; p++)
		if ((points[p] - query).length() < (points[bestIdx] - query).length())
			bestIdx = p;

	return bestIdx;
}

unsigned int findNnKd(const vector<KdNode> &nodes, const vector<Point> &points,
		const Point &query)
{
	// TODO: Implement a recursive kD-Tree nearest neighbor search.

	return 0;
}

__global__ void findNnKd(int* nns, const Point* points, const int numPoints,
		const KdNode* nodes, const int numNodes, const Point* queries,
		const int numQueries)
{
	// TODO: Run one recursive kD-Tree nearest neighbor search per thread.

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > numQueries)
		return;

	nns[idx] = 0;
}

int main(int argc, char **argv)
{
	// Increase the cuda stack size for more recursion levels.
	cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024);

	// Create random point data
	vector<Point> points(NUM_POINTS);
	for (unsigned int p = 0; p < NUM_POINTS; p++)
	{
		points[p].x = randF(-1.0f, 1.0f);
		points[p].y = randF(-1.0f, 1.0f);
	}
	vector<KdNode> nodes = makeKdTree(points);

	vector<Point> queries(NUM_QUERIES);
	for (unsigned int q = 0; q < NUM_QUERIES; q++)
	{
		queries[q].x = randF(-1.0f, 1.0f);
		queries[q].y = randF(-1.0f, 1.0f);
	}

	// Init timing variables
	__int64_t bfTimeCpu = 0;
	__int64_t kdTimeCpu = 0;
	__int64_t kdTimeGpu = 0;
	__int64_t start;

	// BF CPU
#ifdef BF
	vector<int> bfResults(queries.size());
	start = continuousTimeNs();
	for (unsigned int q = 0; q < queries.size(); q++)
		bfResults[q] = findNnBruteForce(points, queries[q]);
	bfTimeCpu += continuousTimeNs() - start;
#endif

	// KD CPU
#ifdef KD
	vector<int> kdResults(queries.size());
	start = continuousTimeNs();
	for (unsigned int q = 0; q < queries.size(); q++)
		kdResults[q] = findNnKd(nodes, points, queries[q]);
	kdTimeCpu += continuousTimeNs() - start;
#endif

	// GPU
#ifdef KD_GPU
	vector<int> kdResultsGpu(queries.size());
	Point* gPoints;
	KdNode* gNodes;
	Point* gQueries;
	int* gNns;
	cudaMalloc(&gPoints, sizeof(Point) * points.size());
	cudaMalloc(&gNodes, sizeof(KdNode) * nodes.size());
	cudaMalloc(&gQueries, sizeof(Point) * queries.size());
	cudaMalloc(&gNns, queries.size() * sizeof(int));
	cudaMemcpy(gPoints, &(points[0]), sizeof(Point) * points.size(),
			cudaMemcpyHostToDevice);
	cudaMemcpy(gNodes, &(nodes[0]), sizeof(KdNode) * nodes.size(),
			cudaMemcpyHostToDevice);
	cudaMemcpy(gQueries, &(queries[0]), sizeof(Point) * queries.size(),
			cudaMemcpyHostToDevice);
	int blocksPerGrid =
			queries.size() % THREADS_PER_BLOCK == 0 ?
					queries.size() / THREADS_PER_BLOCK :
					queries.size() / THREADS_PER_BLOCK + 1;
	start = continuousTimeNs();
	findNnKd<<<blocksPerGrid, THREADS_PER_BLOCK>>>(gNns, gPoints, points.size(),
			gNodes, nodes.size(), gQueries, queries.size());
	cudaDeviceSynchronize();
	kdTimeGpu = continuousTimeNs() - start;
	cudaMemcpy(&(kdResultsGpu[0]), gNns, sizeof(int) * queries.size(),
			cudaMemcpyDeviceToHost);
	cudaFree(gPoints);
	cudaFree(gNodes);
	cudaFree(gQueries);
	cudaFree(gNns);
#endif

	// Verification
	for (unsigned int q = 0; q < queries.size(); q++)
	{
#ifdef VERBOSE
		cout << queries[q] << "   BF: " << bfResults[q] << " "
				<< (queries[q] - points[bfResults[q]]).length() << "   KD: "
				<< kdResults[q] << " "
				<< (queries[q] - points[kdResults[q]]).length() << endl;
#endif
#ifdef BF
#ifdef KD
		if (bfResults[q] != kdResults[q])
			cout << "CPU KD Tree error!" << endl;
#endif
#ifdef KD_GPU
		if (bfResults[q] != kdResultsGpu[q])
			cout << "GPU KD Tree error!" << endl;
#endif
#endif
#ifdef KD
#ifdef KD_GPU
		if (kdResults[q] != kdResultsGpu[q])
			cout << "CPU/GPU KD Tree results differ!" << endl;
#endif
#endif
	}

	// Timing
	cout << "BF time: " << bfTimeCpu << endl;
	cout << "KD time: " << kdTimeCpu << endl;
	cout << "GPU KD time: " << kdTimeGpu << endl;

	return 0;
}
