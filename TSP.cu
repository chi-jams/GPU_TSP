#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
//#include <math.h>

#include <algorithm>

#include "reductions.h"

#define BLOCK_SIZE 32

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
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < Nf) {
        int* perm = d_gen_perm(N, i);        
        
        double dist = 0;
        for (int j = 0; j < N; j++)
            dist += d_get_dist(pts, perm[j], perm[(j+1) % N]);
        dists[i] = dist;

        free(perm);
    }
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
    parallel_tsp(pts, p_min_dist, min_perm, N, Nf);
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

    printf("Welp\n");
    // TODO: Make block nums/block size actually make sense
    calc_paths<<<Nf, 1>>>(d_pts, d_dists, N, Nf);

    printf("Made paths...\n");
    //min_dist = max_reduce(d_dists, N);
    min_perm = 0; // Haven't gotten the actual permutation yet
    printf("Did reduction...\n");

    cudaFree(d_pts);
    cudaFree(d_dists);
}

