#include <cuda_runtime.h>
#include <iostream>
#include <cublas_v2.h>
#include <iomanip>
#include <cmath>
#include <stdlib.h>

#define N 4
#define BLOCK_SIZE 2
#define numStreams 2

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

void printMatrix(double* matrix, const char* name) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%10.4f ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
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

void checkMatrixMultiplication(double* A,double* B,double* C) {
    double* C_ref = new double[N * N]();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
               sum  += A[i + k * N] * B[k + j * N];
             }
             C_ref[i + j * N] = sum;
        }
    }

    double maxError = 0.0;
    double sumError = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double error = std::abs(C_ref[i] - C[i]);
        maxError = std::max(maxError, error);
        sumError += error;
    }
    printMatrix(C_ref, "C_ref");
    delete[] C_ref;

    std::cout << "Max error: " << maxError << std::endl;
    std::cout << "Sum of errors: " << sumError << std::endl;
}



void matrixMultiplyWithStreams(double *A, double *B, double *C) {
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
    //dim3 gridSize(N / BLOCK_SIZE, chunkSize / BLOCK_SIZE);

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

    //cudaDeviceSynchronize();
    int size = N * N * sizeof(double);
    double *h_C = (double *)malloc(size);
    //int size = N * N * sizeof(double);
    double *d_A_c, *d_B_c, *d_C_c;
    cudaMalloc((void **)&d_A_c, size);
    cudaMalloc((void **)&d_B_c, size);
    cudaMalloc((void **)&d_C_c, size);

    cudaMemcpy(d_A_c, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_c, B, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const double alpha = 1.0;
    const double beta = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A_c, N, d_B_c, N, &beta, d_C_c, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "cuBLAS: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_C, d_C_c, size, cudaMemcpyDeviceToHost);

    double maxError = 0.0;
    double sumError = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double error = std::abs(h_C[i] - C[i]);
        maxError = std::max(maxError, error);
        sumError += error;
    }
    std::cout << "Max error: " << maxError << std::endl;
    std::cout << "Sum of errors: " << sumError << std::endl;
    printMatrix(C, "C");
    printMatrix(h_C, "h_c");
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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
    matrixMultiplyWithStreams(h_A, h_B, h_C);

    checkMatrixMultiplication(h_A, h_B, h_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    return 0;
}