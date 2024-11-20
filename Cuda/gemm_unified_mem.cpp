#include <cuda_runtime.h>
#include <iostream>

const int N = 1024;
const int TILE_SIZE = 32;
 
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
    size_t size = N * N * sizeof(double);
    double *h_A, *h_B, *h_C;
    cudaMallocManaged(&h_A, size);
    cudaMallocManaged(&h_B, size);
    cudaMallocManaged(&h_C, size);

    fill_matrix_random_column_major(h_A, N, N);
    fill_matrix_random_column_major(h_B, N, N);
    fill_matrix_C(h_C, N, N);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    cudaEvent_t start, stop;
    float milliseconds;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    matrixMultiply<<<gridSize, blockSize>>>(h_A, h_B, h_C, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;

    // cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}