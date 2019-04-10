
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define uint64_t unsigned long long
#define BLOCK_SIZE 32 
#define blocks_needed(N) ((N / (2 * BLOCK_SIZE)) + (N % (2 * BLOCK_SIZE) == 0 ? 0 : 1))

// Oh yeah, we're introducing macros now
#define sum_reduce_it(BLK_SIZE) if (BLOCK_SIZE >= BLK_SIZE) { \
    if(tid < BLK_SIZE/2) \
        sdata[tid] += sdata[tid+BLK_SIZE/2];__syncthreads(); \
}

#define sum_reduce_warp(BLK_SIZE) if (BLOCK_SIZE >= BLK_SIZE) { \
    sdata[tid] += sdata[tid+BLK_SIZE/2];__syncwarp(); \
}

template <typename T>
__global__ void d_sum_reduce(const T* d_nums, T* d_res, int N) {
    __shared__ T sdata[2 * BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * 2 * BLOCK_SIZE + tid;
    unsigned int gridSize = BLOCK_SIZE * 2 * gridDim.x;

    sdata[tid] = 0;

    // This layers all would-be blocks into a single block
    while (i < N) {
        sdata[tid] += d_nums[i] + d_nums[i + BLOCK_SIZE];
        i += gridSize;
    }
    __syncthreads();

    sum_reduce_it(512);
    sum_reduce_it(256);
    sum_reduce_it(128);
    if (tid < 32) {
        sum_reduce_warp(64);
        sum_reduce_warp(32);
        sum_reduce_warp(16);
        sum_reduce_warp(8);
        sum_reduce_warp(4);
        sum_reduce_warp(2);
    }

    if (tid == 0)
        d_res[blockIdx.x] = sdata[0];
}

template <typename T>
T sum_reduce(const T* nums, int N) {
    unsigned int num_blocks = blocks_needed(N);
    T* d_nums;
    T* d_res;
    
    cudaMalloc(&d_nums, num_blocks * sizeof(T) * 2 * BLOCK_SIZE);
    cudaMalloc(&d_res, num_blocks * sizeof(T));
    cudaMemcpy(d_nums, nums, sizeof(T) * N, cudaMemcpyHostToDevice);

    // TODO: recursive version, for better GPU utilization
    d_sum_reduce<T><<<blocks_needed, BLOCK_SIZE>>>(d_nums, d_res, N);

    T res; 
    cudaMemcpy(&res, d_res, sizeof(T), cudaMemcpyDeviceToHost);
    
    cudaFree(d_nums);
    cudaFree(d_res);

    return res;
}

/* BEGIN: Stuff that should immediately die once we start actually linking
 * things
 */
template <typename T>
T rand_range(T min, T max) {
    double u = rand() / (double)RAND_MAX;
    return (max - min + 1) * u + min;
}

template <typename T>
T* gen_ints(int N) {
    T* nums = (T*) malloc(sizeof(T) * N);
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        nums[i] = rand_range<T>(0, 1000);
    }

    return nums;
}
/* END: Stuff that should die */

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

    uint64_t* nums = gen_ints<uint64_t>(N);

    uint64_t* ind = (uint64_t*) malloc(sizeof(uint64_t) * N);
    
    uint64_t sum = 0;
    for (int i = 0; i < N; i++)
        sum += nums[i];
    printf("Serial Sum: %llu\n", sum);

    uint64_t par_sum = sum_reduce<uint64_t>(nums, N);
    printf("Parallel Sum: %llu\n", par_sum);

    free(nums);
    return 0;
}
