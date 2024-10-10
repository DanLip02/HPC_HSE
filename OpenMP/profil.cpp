#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>


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

void matrixMultiplyBlock(double* A, double* B, double* C, int M, int N, int K, int BLOCK_SIZE) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int j = 0; j < K; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                for (int ii = i; ii < (i + BLOCK_SIZE < M ? i + BLOCK_SIZE : M); ++ii) {
                    for (int jj = j; jj < (j + BLOCK_SIZE < K ? j + BLOCK_SIZE : K); ++jj) {
                        double sum = 0.0;
                        for (int kk = k; kk < (k + BLOCK_SIZE < N ? k + BLOCK_SIZE : N); ++kk) {
                            sum += A[ii * N + kk] * B[kk * K + jj];
                        }
                        C[ii * K + jj] += sum;
                    }
                }
            }
        }
    }
}

void Dgme(int M, int N, int K, double* A, double* B, double* C) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
               sum  += A[i + k * M] * B[k + j * K];
             }
             C[i + j * M] = sum;
        }
    }
}

void Dgme_sched(int M, int N, int K, double* A, double* B, double* C) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
               sum  += A[i + k * M] * B[k + j * K];
             }
             C[i + j * M] = sum;
        }
    }
}

int main() {
    int M = 1500, N = 1500;
    int K = 1500; 

    double start_time_full = omp_get_wtime();
    double* A = new double[M * N];
    double* B = new double[N * K];
    double* C = new double[M * K];

    srand(time(0));

    double start_time_A = omp_get_wtime(); 
    fill_matrix_random_column_major(A, M, N);
    double end_time_A = omp_get_wtime();
    printf("Fill matrix A%f", end_time_A - start_time_A);
    printf("\n");

    double start_time_B = omp_get_wtime(); 
    fill_matrix_random_column_major(B, N, K);
    double end_time_B = omp_get_wtime();
    printf("Fill matrix B%f", end_time_b - start_time_B);
    printf("\n");

    double start_time_C = omp_get_wtime(); 
    fill_matrix_C(C, M, K);
    double end_time_C = omp_get_wtime();
    printf("Fill matrix C%f", end_time_C - start_time_C);
    printf("\n");

    double start_time_cicle = omp_get_wtime(); 
    for (int threads = 1; threads <= 16; threads *= 2) {
        omp_set_num_threads(threads);
        printf("Result with %d threads MKL:\n", threads);
        for (int i = 0; i < 20; i++){
            fill_matrix_C(C, M, K);
            double start_time = omp_get_wtime(); 
            Dgme(M, N, K, A, B, C);
            double end_time = omp_get_wtime();
            printf("%f,", end_time - start_time);
        }
        printf("\n");
    }
    double end_time_cicle = omp_get_wtime();
    printf("cicle%f", end_time_cicle - start_time_cicle);
    printf("\n");
    
    free(A);
    free(B);
    free(C);
    double end_time_full = omp_get_wtime();
    printf("full%f", end_time_full - start_time_full);
    return 0;
}