#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>

#define L 1.0         // Длина стержня
#define U0 1.0        // Начальная температура
#define T1 0.0001     // Конечное время
#define K 1.0         // Коэффициент температуропроводности

double exact_solution(double x, double t) {
    double sum = 0.0;
    for (int m = 0; m < 1000; m++) {
        double term = exp(-K * pow(M_PI, 2) * pow(2 * m + 1, 2) * t / pow(L, 2)) / (2 * m + 1);
        sum += term * sin(M_PI * (2 * m + 1) * x / L);
    }
    return 4 * 1 / M_PI * sum;
}

double model_solution(int N, int rank, int total_ranks, double* rmse_out) {
    double H = L / (N - 1);               // Шаг по пространству
    double TAU = 0.5 * H * H / K;        // Шаг по времени
    int steps = (int)(T1 / TAU);          // Количество временных шагов

    double coeff = K * TAU / (H * H);  // коэффициент
    if (coeff >= 1.0) {
        if (rank == 0) {
            printf("Условие устойчивости нарушено: K * TAU / H^2 >= 1\n");
        }
        MPI_Finalize();
        return 1;
    }

    int base_num = N / total_ranks;
    int remainder = N % total_ranks;
    int local_num = base_num + (rank < remainder ? 1 : 0);

    double* u_current = (double*)malloc(local_num * sizeof(double));
    for (int i = 0; i < local_num; i++) {
        u_current[i] = U0;
    }

    double startTime = MPI_Wtime();

    for (int t = 0; t < steps; t++) {
        double left_boundary = 0.0, right_boundary = 0.0;

        if (rank > 0) {
            MPI_Sendrecv(&u_current[0], 1, MPI_DOUBLE, rank - 1, 0,
                         &left_boundary, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < total_ranks - 1) {
            MPI_Sendrecv(&u_current[local_num - 1], 1, MPI_DOUBLE, rank + 1, 0,
                         &right_boundary, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        double* u_next = (double*)malloc(local_num * sizeof(double));
        for (int i = 0; i < local_num; i++) {
            double left = (i == 0 && rank > 0) ? left_boundary : (i > 0 ? u_current[i - 1] : 0.0);
            double right = (i == local_num - 1 && rank < total_ranks - 1) ? right_boundary : (i < local_num - 1 ? u_current[i + 1] : 0.0);
            u_next[i] = u_current[i] + TAU * K / (H * H) * (left - 2 * u_current[i] + right);
        }

        if (rank == 0) u_next[0] = 0.0;
        if (rank == total_ranks - 1) u_next[local_num - 1] = 0.0;

        free(u_current);
        u_current = u_next;
    }
    double endTime = MPI_Wtime();

    double* u_final = NULL;
    if (rank == 0) {
        u_final = (double*)malloc(N * sizeof(double));
    }

    int* counts = (int*)malloc(total_ranks * sizeof(int));
    int* displs = (int*)malloc(total_ranks * sizeof(int));

    for (int i = 0; i < total_ranks; i++) {
        counts[i] = base_num + (i < remainder ? 1 : 0);
        displs[i] = i * base_num + (i < remainder ? i : remainder);
    }

    MPI_Gatherv(u_current, local_num, MPI_DOUBLE, u_final, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("x\tExact Solution\tApproximate Solution\tAbsolute Error\n");
        //double time = 0.;
        for (int i = 0; i < N; i++) {
            double x = i * L / (N - 1);
            double exact_value = exact_solution(x, T1);
            double approx_value = u_final[i];
            double error = fabs(exact_value - approx_value);
            printf("%.6f\t%.6f\t%.6f\t%.6f\n", x, exact_value, approx_value, error);
        }
    }

    if (rank == 0) {
        double rmse = 0.0;
        for (int i = 0; i < N; i++) {
            double x = i * H;
            double exact = exact_solution(x, T1);
            rmse += pow(u_final[i] - exact, 2);
        }
        rmse = sqrt(rmse / N);
        *rmse_out = rmse;

        free(u_final);
    }

    free(u_current);
    free(counts);
    free(displs);
    return endTime - startTime;
}

int main(int argc, char** argv) {
    int rank, total_ranks;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_ranks);

    // if (argc < 2) {
    //     if (rank == 0) {
    //         fprintf(stderr, "Использование: %s <количество_точек>\n", argv[0]);
    //     }
    //     MPI_Finalize();
    //     return 1;
    // }

    // int N = atoi(argv[1]);
    // if (N < 2) {
    //     if (rank == 0) {
    //         printf(stderr, "Количество точек должно быть >= 2.\n");
    //     }
    //     MPI_Finalize();
    //     return 1;
    // }
    int N = 10000;
    int num_runs = 1;
    double total_time = 0.0;
    double total_rmse = 0.0;

    for (int i = 0; i < num_runs; i++) {
        double rmse = 0.0;
        double time = model_solution(N, rank, total_ranks, &rmse);
        // if (rank == 0) {
        //     total_time += time;
        //     total_rmse += rmse;
        // }
    }

    MPI_Finalize();
    return 0;
}
