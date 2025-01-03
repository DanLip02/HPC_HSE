#include <iostream>
#include <cuda_runtime.h>
const int N = 1024;
const int BLOCK_SIZE = 32;

__global__ void matrixMultiplyShared(double *A, double *B, double *C) {
    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    double sum = 0.0;
    int THREADS_PER_BLOCK = BLOCK_SIZE;
    for (int i = 0; i < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
        //As[threadIdx.y][threadIdx.x] = A[row * N + i * BLOCK_SIZE + threadIdx.x];
        //Bs[threadIdx.y][threadIdx.x] = B[(i * BLOCK_SIZE + threadIdx.y) * N + col];
        int k = i;
        if (row < N && (k * THREADS_PER_BLOCK + threadIdx.x) < N) {
            As[threadIdx.y][threadIdx.x] = A[((k * THREADS_PER_BLOCK) + threadIdx.x) * N + row];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (col < N && (k * THREADS_PER_BLOCK + threadIdx.y) < N) {
            Bs[threadIdx.y][threadIdx.x] = B[col * N + (k * THREADS_PER_BLOCK + threadIdx.y)];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; ++j) {
            sum += As[threadIdx.y][j] * Bs[j][threadIdx.x];
        }

        __syncthreads();
    }

    C[col * N + row] = sum;
}

void printMatrix(double* matrix, const char* name) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%10.4f ", matrix[i * N + j]); // Формат для вывода с точностью до 4 знаков
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


int main() {
    //const int N = 8192;
    //const int BLOCK_SIZE = 32;
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

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaEvent_t start, stop;
    float milliseconds;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matrixMultiplyShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printMatrix(h_C, "shareed");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
