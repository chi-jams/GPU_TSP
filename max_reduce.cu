
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define uint64_t unsigned long long
#define BLOCK_SIZE 32 
#define blocks_needed(N) ((N / (2 * BLOCK_SIZE)) + (N % (2 * BLOCK_SIZE) == 0 ? 0 : 1))

template <typename T>
__global__ void _d_gen_index_arr(T* d_nums, int N) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * BLOCK_SIZE + tid;
    unsigned int gridSize = BLOCK_SIZE * gridDim.x;

    while (i < N) {
        d_nums[i] = i;
        i += gridSize;
    }
}

template <typename T>
T* d_gen_index_arr(int N) {
    T* d_nums; 
    cudaMalloc(&d_nums, sizeof(T) * N);

    _d_gen_index_arr<T><<<1, BLOCK_SIZE>>>(d_nums, N);    

    return d_nums;
}

#define max_reduce_it(BLK_SIZE) if (BLOCK_SIZE >= BLK_SIZE) { \
    if (tid < BLK_SIZE / 2 && sdata[tid+BLK_SIZE/2] > sdata[tid]) { \
        sdata[tid] = sdata[tid+BLK_SIZE/2]; \
    } \
    __syncthreads(); \
}

#define max_reduce_warp(BLK_SIZE) if (BLOCK_SIZE >= BLK_SIZE) { \
    if (sdata[tid+BLK_SIZE/2] > sdata[tid]) { \
        sdata[tid] = sdata[tid+BLK_SIZE/2]; \
    } \
    __syncwarp(); \
}

template <typename T>
__global__ void d_max_reduce(const T* d_nums, T* d_res, int N) {
    __shared__ T sdata[2 * BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * 2 * BLOCK_SIZE + tid;
    unsigned int gridSize = BLOCK_SIZE * 2 * gridDim.x;

    sdata[tid] = 0;

    // This rolls all would-be blocks into a single block
    while (i < N) {
        if (d_nums[i] > d_nums[i + BLOCK_SIZE] && d_nums[i] > sdata[tid]) {
            sdata[tid] = d_nums[i];
        }
        else if (d_nums[i + BLOCK_SIZE] > sdata[tid]) {
            sdata[tid] = d_nums[i + BLOCK_SIZE];
        }
        i += gridSize;
    }
    __syncthreads();

    max_reduce_it(512);
    max_reduce_it(256);
    max_reduce_it(128);
    if (tid < 32) {
        max_reduce_warp(64);
        max_reduce_warp(32);
        max_reduce_warp(16);
        max_reduce_warp(8);
        max_reduce_warp(4);
        max_reduce_warp(2);
    }

    if (tid == 0) {
        d_res[blockIdx.x] = sdata[0];
    }
}

// TODO: Finish converting this to max_reduce
template <typename T>
T max_reduce(const T* nums, int N) {
    unsigned int num_blocks = blocks_needed(N);
    T* d_nums;
    T* d_res;
    
    cudaMalloc(&d_nums, num_blocks * sizeof(T) * 2 * BLOCK_SIZE);
    cudaMalloc(&d_res, num_blocks * sizeof(T));
    cudaMemcpy(d_nums, nums, sizeof(T) * N, cudaMemcpyHostToDevice);

    // TODO: recursive version, for better GPU utilization
    //d_sum_reduce<T><<<blocks_needed, BLOCK_SIZE>>>(d_nums, d_res, N);
    d_max_reduce<T><<<1, BLOCK_SIZE>>>(d_nums, d_res, N);

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
    return (max - min+ 1) * u + min;
}

template <typename T>
T* gen_ints(int N) {
    T* nums = (T*) malloc(sizeof(T) * N);
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        nums[i] = rand_range<T>(0, 100000);
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

    uint64_t max = nums[0];
    uint64_t max_ind = 0;
    for (int i = 1; i < N; i++) {
        if (nums[i] > max) {
            max = nums[i];
            max_ind = i;
        }
    }
    printf("Serial max is %llu at %llu\n", max, max_ind);

    uint64_t par_max = max_reduce<uint64_t>(nums, N);
    printf("Parallel max: %llu\n", par_max);
    
    free(nums);
    return 0;
}

