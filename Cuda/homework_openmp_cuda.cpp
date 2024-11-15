#include <stdlib>
#include <omp.h>
#include <cstdlib>

#define N 1024

void fill_matrix_random_column_major(double* matrix, int rows, int cols) {
    for (int row = 0; row < rows; row++){
        for (int col = 0; col < cols; col++) {
            matrix[col + cols * row] = rand() % 10;
        }
    }
}

int main() {
    double *A, *B, *C;
    A = new double[N * N];
    B = new double[N * N];
    C = new double[N * N];

    fill_matrix_random_column_major(A, N * N);
    fill_matrix_random_column_major(B, N * N);

    double start = omp_get_wtime();

    #pragma omp target data map(to: A[0:N*N], B[0:N*N]) map(from: C[0:N*N])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    double end = omp_get_wtime();
    std::cout << "Elapsed time: " << (end - start) << " seconds\n";

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
