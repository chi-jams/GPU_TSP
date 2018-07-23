
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define BLOCK_SIZE 512 
#define blocks_needed(N) ((N / (2 * BLOCK_SIZE)) + (N % (2 * BLOCK_SIZE) == 0 ? 0 : 1))

template <typename T>
__global__ void d_sum_reduce(const T* d_nums, T* d_res, int N) {
    __shared__ T sdata[2 * BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * 2 * BLOCK_SIZE + tid;
    unsigned int gridSize = BLOCK_SIZE * 2 * gridDim.x;

    sdata[tid] = 0;

    while (i < N) {
        sdata[tid] += d_nums[i] + d_nums[i + BLOCK_SIZE];
        i += gridSize;
    }
    __syncthreads();

    if (BLOCK_SIZE >= 512) {if(tid < 256) sdata[tid] += sdata[tid+256];__syncthreads();}
    if (BLOCK_SIZE >= 256) {if(tid < 128) sdata[tid] += sdata[tid+128];__syncthreads();}
    if (BLOCK_SIZE >= 128) {if(tid <  64) sdata[tid] += sdata[tid+ 64];__syncthreads();}
    // below in one warp
    if (BLOCK_SIZE >= 64) {sdata[tid] += sdata[tid + 32];__syncthreads();}
    if (BLOCK_SIZE >= 32) {sdata[tid] += sdata[tid + 16];__syncthreads();}
    if (BLOCK_SIZE >= 16) {sdata[tid] += sdata[tid +  8];__syncthreads();}
    if (BLOCK_SIZE >=  8) {sdata[tid] += sdata[tid +  4];__syncthreads();}
    if (BLOCK_SIZE >=  4) {sdata[tid] += sdata[tid +  2];__syncthreads();}
    if (BLOCK_SIZE >=  2) {sdata[tid] += sdata[tid +  1];__syncthreads();}

    if (tid == 0)
        d_res[blockIdx.x] = sdata[0];
}

template <typename T>
T sum_reduce(const T* nums, int N) {
    unsigned int num_blocks = blocks_needed(N);
    T* d_nums;
    T* d_res;
    
    cudaMalloc(&d_nums, num_blocks * sizeof(T) * 2 * BLOCK_SIZE);
    cudaMalloc(&d_res, num_blocks * sizeof(int));
    cudaMemcpy(d_nums, nums, sizeof(T) * N, cudaMemcpyHostToDevice);

    d_sum_reduce<T><<<num_blocks, BLOCK_SIZE>>>(d_nums, d_res, N);

    T res; 
    cudaMemcpy(&res, d_res, sizeof(T), cudaMemcpyDeviceToHost);
    
    cudaFree(d_nums);
    cudaFree(d_res);

    return res;
}

int rand_range(int min, int max) {
    double u = rand() / (double)RAND_MAX;
    return (max - min + 1) * u + min;
}

int* gen_ints(int N) {
    int* nums = (int*) malloc(sizeof(int) * N);
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        nums[i] = rand_range(0, 1000);
        //nums[i + 1] = rand_range(0, 1000);
    }

    return nums;
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

    int* nums = gen_ints(N);

    for (int i = 0; i < N; i++)
        printf("%d: %d\n", i, nums[i]);

    unsigned long long sum = 0;
    for (int i = 0; i < N; i++)
        sum += nums[i];
    printf("Serial Sum: %llu\n", sum);

    int par_sum = sum_reduce<int>(nums, N);
    printf("Parallel Sum: %d\n", par_sum);

    free(nums);
    return 0;
}
