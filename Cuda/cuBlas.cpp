#include <iostream>
#include <cublas_v2.h>

#define N 4096  // Размер матриц

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

int main() {
    int size = N * N * sizeof(double);

    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);

    initMatrix(h_A, N);
    initMatrix(h_B, N);

    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const double alpha = 1.0;
    const double beta = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Время выполнения с использованием cuBLAS: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
