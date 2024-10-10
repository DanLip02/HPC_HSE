#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
//#include <cblas.h>
#include <omp.h>
#include <time.h>

void fill_matrix_random(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() % 10; 
    }
}

void fill_matrix_C(double* matrix, int rows, int cols) {
    for (int row = 0; row < rows; row++){
        for (int col = 0; col < cols; col++) {
            matrix[col + cols * row] = 0.;
        }
    }
}

int compare_matrices(int M, int K, double* A, double* B, double eps) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            double diff = fabs(A[i * K + j] - B[i * K + j]);
            if (diff > tolerance) {
                printf("Difference at element (%d, %d): A = %f, B = %f\n", i, j, A[i * K + j], B[i * K + j]);
                return 0;
            }
        }
    }
    return 1; 
}

int main() {
    int M = 1500, N = 1500, K = 1500; 
    double* A = new double[M * N];
    double* B = new double[N * K];
    double* C = new double[M * K];

    srand(time(0));
    fill_matrix_random(A, M, N);
    fill_matrix_random(B, N, K);

    //for (int threads = 1; threads <= 24; threads++)
    for (int threads = 1; threads <= 16; threads *= 2) {
        //omp_set_num_threads(threads); //no dif between mkl or omp set nums
        //openblas_set_num_threads(threads); //for cblas
        mkl_set_num_threads(threads);
        printf("Result with %d threads MKL:\n", threads);
        for (int step = 0; step <= 100; step ++){
            fill_matrix_C(C, M, K);
            double start_time_MKL = omp_get_wtime();  
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                        M, K, N, 
                        1.0, A, M, 
                        B, N, 
                        0.0, C, M);
                        
            double end_time_MKL = omp_get_wtime();
            double time_taken_MKL = end_time_MKL - start_time_MKL;
            printf("%f,", time_taken);
        }
        printf("\n");
    }
    free(A);
    free(B);
    free(C);
    
    return 0;
}


