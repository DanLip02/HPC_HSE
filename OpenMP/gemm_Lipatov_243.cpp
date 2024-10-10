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

void Dgme_block(int M, int N, int K, double* A, double* B, double* C, int BLOCK_SIZE) {
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
            //#pragma omp simd
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
    double* A = new double[M * N];
    double* B = new double[N * K];
    double* C = new double[M * K];

    srand(time(0));
    fill_matrix_random_column_major(A, M, N);
    fill_matrix_random_column_major(B, N, K);

    for (int threads = 1; threads <= 16; threads *= 2) {
        omp_set_num_threads(threads);
        printf("Result with %d threads MKL:\n", threads);
        for (int i = 0; i < 50; i++){
            fill_matrix_C(C, M, K);
            double start_time = omp_get_wtime(); 
            //Dgme(M, N, K, A, B, C);
            Dgme_block(M, N, K, A, B, C, 32);
            double end_time = omp_get_wtime();
            // printf("Time taken: %f seconds\n\n", end_time - start_time);
            printf("%f,", end_time - start_time);
        }
        // printf("Result with %d threads:\n", threads);
        printf("\n");
    }

    free(A);
    free(B);
    free(C);
    return 0;
}