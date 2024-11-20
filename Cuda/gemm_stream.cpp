#include <cuda_runtime.h>
#include <iostream>

#define N 4096
#define BLOCK_SIZE 16
#define numStreams 4

__global__ void matrixMultiplyKernel(double* A, double* B, double* C, int chunkSize) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    double sum = 0.0;
    if (row < N && col < chunkSize) {
        for (int i = 0; i < N; ++i) {
            sum += A[row +  N * i] * B[i +  N * col];
        }
        C[col * N + row] = sum;
    }
}

void fill_matrix_random_column_major(double* matrix, int rows, int cols) {
    for (int row = 0; row < rows; row++){
        for (int col = 0; col < cols; col++) {
            matrix[col + cols * row] = rand() % 10;
        }
    }
}

void fill_matrix_C(double* matrix, int rows, int cols) {
    for (int row = 0; row < rows; row++){
        for (int col = 0; col < cols; col++) {
            matrix[col + cols * row] = 0.;
        }
    }
}

void matrixMultiplyCUDAStreams(double *A, double *B, double *C) {
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * N * sizeof(double));
    cudaMalloc((void **)&d_B, N * N * sizeof(double));
    cudaMalloc((void **)&d_C, N * N * sizeof(double));

    cudaStream_t streams[numStreams];

    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    int chunkSize = N / numStreams;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // dim3 gridSize(N / BLOCK_SIZE, chunkSize / BLOCK_SIZE);

    for (int i = 0; i < numStreams; ++i) {
        int offset = i * chunkSize * N;
        cudaMemcpyAsync(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_B + offset, B + offset, chunkSize * N * sizeof(double), cudaMemcpyHostToDevice, streams[i]);

        matrixMultiplyKernel<<<gridSize, blockSize, 0, streams[i]>>>(d_A, d_B + offset, d_C + offset, chunkSize);

        cudaMemcpyAsync(C + offset, d_C + offset, chunkSize * N * sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    double *h_A = new double[N * N];
    double *h_B = new double[N * N];
    double *h_C = new double[N * N];

    fill_matrix_random_column_major(h_A, N, N);
    fill_matrix_random_column_major(h_B, N, N);
    matrixMultiplyCUDAStreams(h_A, h_B, h_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    return 0;
}