#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <float.h>
//#include <math.h>

#include <algorithm>

#define BLOCK_SIZE 32
#define blocks_needed(N) ((N / (2 * BLOCK_SIZE)) + (N % (2 * BLOCK_SIZE) == 0 ? 0 : 1))

int rand_range(int min, int max);
int* make_pts(int N);
int* gen_perm(int n, int k);

double get_dist(const int* pts, int i, int j);

void serial_tsp(const int* pts, double& min_dist, int& min_perm, int N, int Nf);
void parallel_tsp(const int* pts, double& min_dist, int& min_perm, int N, int Nf);

__device__ int* d_gen_perm(int n, int perm) {
    int i, ind, m=perm;
    int* p = (int*) malloc(sizeof(int) * n);
    int* e = (int*) malloc(sizeof(int) * n);

    for (i=0;i<n;i++)
        e[i]=i;
    for (i=0;i<n;i++) {
        ind = m % (n - i);
        m = m / (n - i);
        p[i] = e[ind];
        e[ind] = e[n - i - 1];
    }

    free(e);

    return p;
}

__device__ double d_get_dist(const int* pts, int i, int j) {
    int x1 = pts[2*i],
        y1 = pts[2*i + 1],
        x2 = pts[2*j],
        y2 = pts[2*j + 1];

    int dx = x1 - x2,
        dy = y1 - y2;
    dx *= dx;
    dy *= dy;
    
    return sqrtf(dx + dy);
}

__global__ void calc_paths(const int* pts, double* dists, int N, int Nf) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    //unsigned int gridSize = BLOCK_SIZE * gridDim.x;

    if (i < Nf) {
        int* perm = d_gen_perm(N, i);        
        
        double dist = 0;
        for (int j = 0; j < N; j++)
            dist += d_get_dist(pts, perm[j], perm[(j+1) % N]);
        dists[i] = dist;
        //printf("dist %d: %f\n", i, dist);

        free(perm);
        //i += gridSize;
    }
}

#define min_reduce_it(BLK_SIZE) if (BLOCK_SIZE >= BLK_SIZE) { \
    if (tid < BLK_SIZE / 2 && sdata[tid+BLK_SIZE/2] < sdata[tid]) { \
        sdata[tid] = sdata[tid+BLK_SIZE/2]; \
    } \
    __syncthreads(); \
}

#define min_reduce_warp(BLK_SIZE) if (BLOCK_SIZE >= BLK_SIZE) { \
    if (sdata[tid+BLK_SIZE/2] < sdata[tid]) { \
        sdata[tid] = sdata[tid+BLK_SIZE/2]; \
    } \
    __syncwarp(); \
}

template <typename T>
__global__ void d_min_reduce(const T* d_nums, T* d_res, int N) {
    __shared__ T sdata[2 * BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * 2 * BLOCK_SIZE + tid;
    unsigned int gridSize = BLOCK_SIZE * 2 * gridDim.x;

    sdata[tid] = DBL_MAX;

    // This rolls all would-be blocks into a single block
    while (i < N) {
        if (d_nums[i] < d_nums[i + BLOCK_SIZE] && d_nums[i] < sdata[tid]) {
            sdata[tid] = d_nums[i];
        }
        else if (d_nums[i + BLOCK_SIZE] < sdata[tid]) {
            sdata[tid] = d_nums[i + BLOCK_SIZE];
        }
        i += gridSize;
    }
    __syncthreads();

    min_reduce_it(512);
    min_reduce_it(256);
    min_reduce_it(128);
    if (tid < 32) {
        min_reduce_warp(64);
        min_reduce_warp(32);
        min_reduce_warp(16);
        min_reduce_warp(8);
        min_reduce_warp(4);
        min_reduce_warp(2);
    }

    if (tid == 0) {
        d_res[blockIdx.x] = sdata[0];
    }
}

// TODO: Finish converting this to min_reduce
template <typename T>
T min_reduce(const T* nums, int N) {
    unsigned int num_blocks = blocks_needed(N);
    T* d_nums;
    T* d_res;
    
    cudaMalloc(&d_nums, num_blocks * sizeof(T) * 2 * BLOCK_SIZE);
    cudaMalloc(&d_res, num_blocks * sizeof(T));
    cudaMemset(d_nums, 127, num_blocks * sizeof(T) * 2 * BLOCK_SIZE);
    cudaMemcpy(d_nums, nums, sizeof(T) * N, cudaMemcpyHostToDevice);

    // TODO: recursive version, for better GPU utilization
    //d_sum_reduce<T><<<blocks_needed, BLOCK_SIZE>>>(d_nums, d_res, N);
    d_min_reduce<T><<<1, BLOCK_SIZE>>>(d_nums, d_res, N);

    T res; 
    cudaMemcpy(&res, d_res, sizeof(T), cudaMemcpyDeviceToHost);
    
    cudaFree(d_nums);
    cudaFree(d_res);

    return res;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s N\n", argv[0]);
        exit(-1);
    }

    // Specify the size of the random set of points
    long int N = strtol(argv[1], NULL, 10);
    if (N <= 0) {
        printf("Please enter a positive int for N\n");
        exit(-1);
    }
    else if (N == LONG_MAX || N == LONG_MIN) {
        printf("The provided N is too %s.\n", N == LONG_MAX ? "large" : "small");
        exit(-1);
    }

    printf("N: %d\n", N);
    // Nf == N!
    unsigned long long int Nf = 1;
    for (int i=N;i>0;i--) Nf*=i;

    int* pts = make_pts(N);

    // Print all of the points
    for (int i = 0; i < 2 * N; i+=2)
        printf("%d: (%d, %d)\n", i / 2, pts[i], pts[i + 1]);

    /* Serial TSP calculation */
    double min_dist; // The length of the smallest distance
    int min_perm;    // The number of the smallest permutation
    serial_tsp(pts, min_dist, min_perm, N, Nf);
    printf("Min distance: %f\n", min_dist);
    printf("Path:\n");
    int* perm = gen_perm(N, min_perm);
    for (int i = 0; i < N; i++) {
        int p_i = perm[i];
        printf("%d: (%d, %d)\n", p_i, pts[2*p_i], pts[2*p_i + 1]);
    }
    /* End Serial TSP calculation */

    /* Parallel TSP calculation */
    double p_min_dist;
    int p_min_perm;
    parallel_tsp(pts, p_min_dist, p_min_perm, N, Nf);
    printf("Parallel min distance: %f\n", p_min_dist);
    /* End Parallel TSP calculation */


    free(pts);
    return 0;
}

// Bounds inclusive
int rand_range(int min, int max) {
    double u = rand() / (double)RAND_MAX;
    return (max - min + 1) * u + min;
}

// Points are stored in a 1-D array of size 2*N
int* make_pts(int N) {
    int* pts = (int*) malloc(sizeof(int) * N * 2);
    srand(time(NULL));
    for (int i = 0; i < 2 * N; i+=2) {
        pts[i] = rand_range(0, 1000);
        pts[i + 1] = rand_range(0, 1000);
    }

    return pts;
}

// Generates the perm'th permutation for an array of size n
// perm can be an int [0, n!)
int* gen_perm(int n, int perm) {
    int i, ind, m=perm;
    int* p = (int*) malloc(sizeof(int) * n);
    int* e = (int*) malloc(sizeof(int) * n);

    for (i=0;i<n;i++)e[i]=i;
    for (i=0;i<n;i++) {
        ind = m % (n - i);
        m = m / (n - i);
        p[i] = e[ind];
        e[ind] = e[n - i - 1];
    }

    free(e);

    return p;
}

double get_dist(const int* pts, int i, int j) {
    int x1 = pts[2*i],
        y1 = pts[2*i + 1],
        x2 = pts[2*j],
        y2 = pts[2*j + 1];

    int dx = x1 - x2,
        dy = y1 - y2;
    dx *= dx;
    dy *= dy;
    
    return sqrt(dx + dy);
}

// Brute-force TSP solver, serial version. Results are returned in min_dist and
// min_perm.
void serial_tsp(const int* pts, double& min_dist, int& min_perm, int N, int Nf) {
    min_dist = -1;
    int* perm;

    for (int i = 0; i < Nf; i++) {
        perm = gen_perm(N, i); 

        double dist = 0;
        for (int j = 0; j < N; j++) {
            dist += get_dist(pts, perm[j], perm[(j+1) % N]);
        }

        if (dist < min_dist || min_dist < 0) {
            min_dist = dist;
            min_perm = i;
        }
        
        free(perm);
    }
}

void parallel_tsp(const int* pts, double& min_dist, int& min_perm, int N, int Nf) {
    int* d_pts;
    double* d_dists;
    cudaMalloc(&d_pts, 2 * N * sizeof(int));
    cudaMalloc(&d_dists, Nf * sizeof(double));

    printf("Malloc'd\n");
    cudaMemcpy(d_pts, pts, 2 * N * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < 2 * N; i += 2)
        printf("Real: %d %d\n", pts[i], pts[i + 1]);

    printf("Copied points\n");
    // TODO: Make block nums/block size actually make sense
    calc_paths<<<1, BLOCK_SIZE>>>(d_pts, d_dists, N, Nf);

    double* why;
    why = (double*) malloc(sizeof(double) * Nf);
    cudaMemcpy(why, d_dists, sizeof(double) * Nf, cudaMemcpyDeviceToHost);

    printf("Made paths...\n");
    min_dist = min_reduce<double>(why, Nf);
    min_perm = 0; // Haven't gotten the actual permutation yet
    printf("Did reduction...\n");

    cudaFree(d_pts);
    cudaFree(d_dists);
}
