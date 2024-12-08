#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define L 1.0      // длина стержня
#define K 1.0      // коэффициент температуропроводности
#define H 0.01     // шаг по пространству
#define TAU 0.00005 // шаг по времени
#define T 0.1  // время моделирования
#define N ((int)(L / H) + 1) //округление вверх

// #define cudaCheckError(call)                                                  
//     do {                                                                      
//         cudaError_t err = call;                                               
//         if (err != cudaSuccess) {                                             
//             fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  
//                     cudaGetErrorString(err));                                 
//             exit(EXIT_FAILURE);                                               
//         }                                                                     
//     } while (0)

__global__ void heat_transfer_step(double* u_prev, double* u_curr, int n, double mnozh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < n - 1) {
        u_curr[idx] = u_prev[idx] + mnozh * (u_prev[idx + 1] - 2 * u_prev[idx] + u_prev[idx - 1]);
    }
}

double exact_solution(double x, double t, double k, double l, int max_terms) {
    double sum = 0.0;
    for (int m = 0; m < max_terms; m++) {
        double coeff = (2.0 * m + 1.0);
        double exp_term = exp(-k * M_PI * M_PI * coeff * coeff * t / (l * l));
        double sin_term = sin(M_PI * coeff * x / l);
        sum += exp_term * sin_term / coeff;
    }
    return (4.0 / M_PI) * sum;
}
int main(int argc, char** argv) {
    double mnozh = (K * TAU / (H * H));

    if (mnozh < 1) {
        int rank, size;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (size != 2) {
            if (rank == 0) {
                printf("Ошибка: для гибридной задачи должно быть ровно 2 MPI-процесса.\n");
            }
            MPI_Finalize();
            return -1;
        }

        int no_cash = N / size;
        int cash = N % size;
        int points_per_proc = (rank < cash) ? no_cash + 1 : no_cash;

        double *u_prev, *u_curr, *d_u_prev, *d_u_curr;
        cudaMallocHost(&u_prev, (points_per_proc + 2) * sizeof(double));
        cudaMallocHost(&u_curr, (points_per_proc + 2) * sizeof(double));
        cudaMalloc(&d_u_prev, (points_per_proc + 2) * sizeof(double));
        cudaMalloc(&d_u_curr, (points_per_proc + 2) * sizeof(double));

        double *change_prev = u_prev;
        double *change_curr = u_curr;

        for (int i = 0; i <= points_per_proc + 1; i++) {
            u_prev[i] = 1.0;
        }
        u_prev[0] = 0.0;
        u_prev[points_per_proc + 1] = 0.0;
        if (rank == 0){
            u_prev[1] = 0.0;
        }
        if (rank == size-1){
            u_prev[points_per_proc] = 0.0;
        }

        double start_time = MPI_Wtime();

        for (double t = 0; t <= T_MAX; t += TAU) {
            if (rank > 0) {
                MPI_Send(&u_prev[1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&u_prev[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (rank < size - 1) {
                MPI_Recv(&u_prev[points_per_proc + 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&u_prev[points_per_proc], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            }
            cudaMemcpy(d_u_prev, u_prev, (points_per_proc + 2) * sizeof(double), cudaMemcpyHostToDevice);
            heat_transfer_step<<<(points_per_proc + 31) / 32, 32>>>(d_u_prev, d_u_curr, points_per_proc + 2, mnozh);
            cudaMemcpy(u_curr, d_u_curr, (points_per_proc + 2) * sizeof(double), cudaMemcpyDeviceToHost);

            if (rank == 0){
                u_curr[1] = 0.0;
                u_prev[1] = 0.0;
            }
            if (rank == size-1){
                u_curr[points_per_proc] = 0.0;
                u_prev[points_per_proc] = 0.0;
            }

            double *temp = u_prev;
            u_prev = u_curr;
            u_curr = temp;
        }
        cudaCheckError(cudaDeviceSynchronize());
        double end_time = MPI_Wtime();

        double *result = NULL;
        int *recvcounts = NULL;
        int *displs = NULL;

        if (rank == 0) {
            result = (double *)malloc(N * sizeof(double));
            recvcounts = (int *)malloc(size * sizeof(int));
            displs = (int *)malloc(size * sizeof(int));

            for (int i = 0; i < size; i++) {
                recvcounts[i] = (i < cash) ? no_cash + 1 : no_cash;
                displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
            }
        }

        MPI_Gatherv(&u_prev[1], points_per_proc, MPI_DOUBLE,
                    result, recvcounts, displs, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("MPI решение:\n");
            for (int i = 0; i < N; i++) {
                printf("x=%.2f, u=%.5f\n", i * H, result[i]);
            }
            printf("Время выполнения: %.5f секунд\n", end_time - start_time);

            int n = 101;
            double h = L / (n - 1);
            int max_terms = 100;
            printf("Точное решение:\n");
            for (int i = 0; i < n; i++) {
                double x = i * h; // x координата
                double u = exact_solution(x, T_MAX, K, L, max_terms);
                printf("x=%.2f, u=%.5f     %.5f\n", x, u, fabs(result[i] - u));
            }

            free(result);
            free(recvcounts);
            free(displs);
        }

        cudaFreeHost(change_prev);
        cudaFreeHost(change_curr);
        cudaFree(d_u_prev);
        cudaFree(d_u_curr);
        MPI_Finalize();
    } else {
        printf("Не выполняется условие!");
    }
    return 0;
}