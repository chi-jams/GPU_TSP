
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

#define min_reduce_it(BLK_SIZE, sync_mtd) if (BLOCK_SIZE >= BLK_SIZE) { \
    if (sdata[tid+BLK_SIZE/2] < sdata[tid]) { \
        sdata[tid] = sdata[tid+BLK_SIZE/2]; \
    } \
    sync_mtd; \
}

template <typename T>
__global__ void d_min_reduce(const T* d_nums, T* d_res, int N) {
    __shared__ T sdata[2 * BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * 2 * BLOCK_SIZE + tid;
    unsigned int gridSize = BLOCK_SIZE * 2 * gridDim.x;

    sdata[tid] = 0;

    // This rolls all would-be blocks into a single block
    while (i < N) {
        if (d_nums[i] < d_nums[i + BLOCK_SIZE] && d_nums[i] < sdata[tid]) {
            sdata[tid] = d_nums[i];
            sind[tid] = d_index[i];
        }
        else if (d_nums[i + BLOCK_SIZE] < sdata[tid]) {
            sdata[tid] = d_nums[i + BLOCK_SIZE];
            sind[tid] = d_index[i + BLOCK_SIZE];
        }
        i += gridSize;
    }
    __syncthreads();

    min_reduce_it(512, __syncthreads());
    min_reduce_it(256, __syncthreads());
    min_reduce_it(128, __syncthreads());
    if (tid < 32) {
        min_reduce_it(64, __syncwarp());
        min_reduce_it(32, __syncwarp());
        min_reduce_it(16, __syncwarp());
        min_reduce_it(8, __syncwarp());
        min_reduce_it(4, __syncwarp());
        min_reduce_it(2, __syncwarp());
    }

    if (tid == 0) {
        d_res[blockIdx.x] = sdata[0];
    }
}

// TODO: Finish converting this to min_reduce
template <typename T>
T min_reduce(const T* nums, int N) {

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

    for (int i = 0; i < N; i++)
        printf("%d: %llu\n", i, nums[i]);

    uint64_t min = nums[0];
    uint64_t min_ind = 0;
    for (int i = 1; i < N; i++) {
        if (nums[i] < min) {
            min = nums[i];
            min_ind = i;
        }
    }
    printf("Serial min is %llu at %llu\n", min, min_ind);
    
    free(nums);
    return 0;
}

