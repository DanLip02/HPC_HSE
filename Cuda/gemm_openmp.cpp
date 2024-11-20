#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

void fill_matrix_random_column_major(double* matrix, int rows, int cols) {
    for (int row = 0; row < rows; row++){
        for (int col = 0; col < cols; col++) {
            matrix[col + cols * row] = rand() % 10;
        }
    }
}

iint main() {
    for (int N = 1024; N <= 8192; N *= 2) {
        int size = N * N * sizeof(double);
        double *A, *B, *C;
        A =  (double *)malloc(size);
        B =  (double *)malloc(size);
        fill_matrix_random_column_major(A, N, N);
        fill_matrix_random_column_major(B, N,  N);
        for(int i = 0; i < 10; i++){
            C =  (double *)malloc(size);
            double start = omp_get_wtime();
            #pragma omp target data map(to: A[0:N*N], B[0:N*N]) map(from: C[0:N*N])
            {
                #pragma omp target teams distribute parallel for collapse(2)
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < N; k++) {
                        sum  += A[i + k * N] * B[k + j * N];
                        }
                        C[i + j * N] = sum;
                    }
                }
            }
            double end = omp_get_wtime();
            printf("%f,",(end - start) * 1000);
            free(C);
        }
        printf("\n");
        free(A);
        free(B);
        //free(C);

    }
    return 0;
}


// for (int BLOCK_ = 1; BLOCK_ <= BLOCK_SIZE; BLOCK_ *= 2){
//     printf("BLOCK %d \n", BLOCK_);
//     for (int i = 1; i < 5; i++){
//        C =  (double *)malloc(size);
//        //printf("BLOCK %d \n", BLOCK_);
//        double start = omp_get_wtime();
//        int NUM_TEAMS = N / BLOCK_;
//        #pragma omp target data map(to: A[0:N*N], B[0:N*N]) map(from: C[0:N*N])
//       {
//          #pragma omp target teams distribute parallel for collapse(2)
//          //#pragma omp target teams num_teams(NUM_TEAMS) distribute parallel for collapse(2)
//          for (int i = 0; i < N; i++) {
//              for (int j = 0; j < N; j++) {
//                  double sum = 0.0;
//                  for (int k = 0; k < N; k++) {
//                  sum  += A[i + k * N] * B[k + j * N];
//                  }
//                  C[i + j * N] = sum;
//              }
//          }
//       }
//       double end = omp_get_wtime();
//       printf("%f,",(end - start) * 1000);
//       free(C);
//       }
//       //printf(
//       //double end = omp_get_wtime();
//       //printf("Elapsed time: %f",(end - start) * 1000);
//       printf("\n");
//       //free(C);
//     }
