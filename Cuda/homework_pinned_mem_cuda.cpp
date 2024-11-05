#include <iostream>
#include <cuda_runtime.h>

__global__ void matrixMultiply(const double *A, const double *B, double *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            sum += A[k * N + row] * B[col * N + k];
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


int main() {
    const int N = 1024;
    size_t size = N * N * sizeof(double);

    double *h_A, *h_B, *h_C;
    cudaMallocHost((void **)&h_A, size);
    cudaMallocHost((void **)&h_B, size);
    cudaMallocHost((void **)&h_C, size);

    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    fill_matrix_random_column_major(h_A, N, N);
    fill_matrix_random_column_major(h_B, N, N);
    fill_matrix_C(h_C, N, N);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(N / 16, N / 16);
    cudaEvent_t start, stop;
    float milliseconds;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
}