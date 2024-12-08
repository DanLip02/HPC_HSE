#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define L 1.0     // Длина стержня
#define K 1.0     // Коэффициент теплопроводности
#define H 0.02    // Шаг по пространству
#define TAU 0.0002 // Шаг по времени
#define T 0.1     // Время моделирования

double exact_solution(double x, double t, int m_terms) {
    double u = 0.0;
    for (int m = 0; m < m_terms; m++) {
        double coef = pow(-1, m) / (2 * m + 1) * exp(-K * pow(M_PI * (2 * m + 1), 2) * t / pow(L, 2));
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

    double* u_current = (double*)malloc(local_nx * sizeof(double));

    double Num = local_nx;
    double steps = nt;
    for (int i = 0; i < Num; i++) {
        u_current[i] = 1.;
    }

    for (int n = 0; n < steps; n++) {
        double left_condition = 0.0, right_condition = 0.0;
        

        if (rank != 0) {
            MPI_Sendrecv(&u_current[0], 1, MPI_DOUBLE, rank - 1, 0, &left_condition, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        if (rank != total_ranks - 1) {
            MPI_Sendrecv(&u_current[Num-1], 1, MPI_DOUBLE, rank + 1, 0, &right_condition, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }


        double *u_next = (double*)malloc((Num) * sizeof(double));
        for (int i = 0; i < Num; i++) {
            double left = 0.0, right = 0.0;

            if (i == 0) {
                left = (rank == 0) ? T : left_condition;
            } else {
                left = u_current[i - 1];
            }

            if (i == Num - 1) {
                right = (rank == total_ranks - 1) ? T : right_condition;
            } else {
                if (Num > 1){
                    right = u_current[i + 1];
                } else {
                    right = (rank == total_ranks - 1) ? T : right_condition;
                }
            }
        
            if (i == 0 && rank == 0) {
                u_next[0] = 0.0;
            } else if (i == Num - 1 && rank == total_ranks - 1) {
                u_next[Num - 1] = 0.0;
            } else {
                u_next[i] = u_current[i] + coeff * (left - 2 * u_current[i] + right);
            }
        }

        free(u_current);
        u_current = u_next;
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

    MPI_Gatherv(u_current, local_nx, MPI_DOUBLE, u_global, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("x\tExact Solution\tApproximate Solution\tAbsolute Error\n");
        double time = 0.;
        for (int i = 0; i < nx; i++) {
            double x = i * L / (nx - 1);
            double exact_value = exact_solution(x, T, 100);
            double approx_value = u_global[i * nx / (nx - 1)]; // интерполяция для 11 точек
            double error = fabs(exact_value - approx_value);
            time += 1 / (T / TAU);
            printf("%.6f\t%.6f\t%.6f\t%.6f\n", x, exact_value, approx_value, error);
        }
    }

    free(u_cuurent);
    if (rank == 0) {
        free(u_global);
    }
    MPI_Finalize();
    return 0;
}
