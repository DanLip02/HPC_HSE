#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define L 1.0     // Длина стержня
#define K 1.0     // Коэффициент теплопроводности
#define H 0.0001    // Шаг по пространству
#define TAU 0.000000001 // Шаг по времени
#define T 0.1     // Время моделирования

double exact_solution(double x, double t, int m_terms) {
    double u = 0.0;
    for (int m = 0; m < m_terms; m++) {
        double coef = exp(-K * pow(M_PI * (2 * m + 1), 2) * t / pow(L, 2)) / (2 * m + 1);
        u += coef * sin(M_PI * (2 * m + 1) * x / L);
    }
    return 4 / M_PI * u;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nx = (int)(L / H) + 1;  // Количество точек по пространству
    int nt = (int)(T / TAU);    // Количество временных шагов
    double ratio = K * TAU / (H * H);

    if (ratio >= 1.0) {
        if (rank == 0) {
            printf("Условие устойчивости не выполняется: K * TAU / H^2 >= 1\n");
        }
        MPI_Finalize();
        return 1;
    }

    int points_per_proc = nx / size;
    int remainder = nx % size;
    int local_nx = (rank < remainder) ? points_per_proc + 1 : points_per_proc;
    int start_idx = (rank < remainder) ? rank * local_nx : rank * points_per_proc + remainder;
    int end_idx = start_idx + local_nx;

    double* u_local = (double*)malloc(local_nx * sizeof(double));
    for (int i = 0; i < local_nx; i++) {
        u_local[i] = 1.0; // Начальная температура
    }
    double start_time = MPI_Wtime();
    for (int n = 0; n < nt; n++) {
        double u_left = 0.0, u_right = 0.0;

        if (rank > 0) {
            MPI_Sendrecv(&u_local[0], 1, MPI_DOUBLE, rank - 1, 0, &u_left, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank < size - 1) {
            MPI_Sendrecv(&u_local[local_nx - 1], 1, MPI_DOUBLE, rank + 1, 1, &u_right, 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        double* u_new = (double*)malloc(local_nx * sizeof(double));
        for (int i = 1; i < local_nx - 1; i++) {
            u_new[i] = u_local[i] + ratio * (u_local[i - 1] - 2 * u_local[i] + u_local[i + 1]);
        }

        if (rank > 0) {
            u_new[0] = u_local[0] + ratio * (u_left - 2 * u_local[0] + u_local[1]);
        }
        if (rank < size - 1) {
            u_new[local_nx - 1] = u_local[local_nx - 1] + ratio * (u_local[local_nx - 2] - 2 * u_local[local_nx - 1] + u_right);
        }

        free(u_local);
        u_local = u_new;
    }

    double end_time = MPI_Wtime();
    double exec_time = end_time - start_time;
    printf("%f \n", exec_time);
    int* recvcounts = NULL;
    int* displs = NULL;
    double* u_global = NULL;
    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        u_global = (double*)malloc(nx * sizeof(double));
        for (int p = 0; p < size; p++) {
            recvcounts[p] = (p < remainder) ? points_per_proc + 1 : points_per_proc;
            displs[p] = (p == 0) ? 0 : displs[p - 1] + recvcounts[p - 1];
        }
    }
    //MPI_Gather(u_local, local_nx, MPI_DOUBLE, u_global, points_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Gatherv(u_local, local_nx, MPI_DOUBLE, u_global, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("x\tExact Solution\tApproximate Solution\tAbsolute Error\n");
        double time = 0.;
        for (int i = 0; i < nx; i++) {
            double x = i * L / (N - 1);
            double exact_value = exact_solution(x, T, 100);
            double approx_value = u_global[i];
            double error = fabs(exact_value - approx_value);
            time += 1 / (T / TAU);
            printf("%.6f\t%.6f\t%.6f\t%.6f\n", x, exact_value, approx_value, error);
        }
    }

    free(u_local);
    if (rank == 0) {
        free(u_global);
    }
    MPI_Finalize();
    return 0;
}