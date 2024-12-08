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
    for (int m = 0; m < 50000; m++) {
        double term = exp(-K * pow(M_PI, 2) * pow(2 * m + 1, 2) * t / pow(L, 2)) / (2 * m + 1);
        sum += term * sin(M_PI * (2 * m + 1) * x / L);
    }
    return 4 * U0 / M_PI * sum;
}

int main(int argc, char** argv) {
    int rank, total_ranks;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_ranks);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Использование: %s <количество_точек>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    if (N < 2) {
        if (rank == 0) {
            fprintf(stderr, "Количество точек должно быть >= 2.\n");
        }
        MPI_Finalize();
        return 1;
    }

    double H = L / (N - 1);               // Шаг по пространству
    double TAU = 0.5 * H * H / K;        // Шаг по времени (для устойчивости)
    int steps = (int)(T1 / TAU);          // Количество временных шагов

    int base_num = N / total_ranks;
    int remainder = N % total_ranks;
    int local_num = base_num + (rank < remainder ? 1 : 0);

    double* initial_temperatures = NULL;
    if (rank == 0) {
        initial_temperatures = (double*)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) {
            initial_temperatures[i] = U0;
        }
    }

    double* local_temperatures = (double*)malloc(local_num * sizeof(double));
    int* counts = (int*)malloc(total_ranks * sizeof(int));
    int* displs = (int*)malloc(total_ranks * sizeof(int));
    for (int i = 0; i < total_ranks; i++) {
        counts[i] = base_num + (i < remainder ? 1 : 0);
        displs[i] = i * base_num + (i < remainder ? i : remainder);
    }
    MPI_Scatterv(initial_temperatures, counts, displs, MPI_DOUBLE, local_temperatures, local_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double coeff = K * TAU / (H * H);
    for (int t = 0; t < steps; t++) {
        double* next_temperatures = (double*)malloc(local_num * sizeof(double));
        for (int i = 0; i < local_num; i++) {
            double left = (i > 0) ? local_temperatures[i - 1] : 0.0;
            double right = (i < local_num - 1) ? local_temperatures[i + 1] : 0.0;
            next_temperatures[i] = local_temperatures[i] + coeff * (left - 2 * local_temperatures[i] + right);
        }
        free(local_temperatures);
        local_temperatures = next_temperatures;
    }

    double* final_temperatures = NULL;
    if (rank == 0) {
        final_temperatures = (double*)malloc(N * sizeof(double));
    }
    MPI_Gatherv(local_temperatures, local_num, MPI_DOUBLE, final_temperatures, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double rmse = 0.0;
        for (int i = 0; i < N; i++) {
            double x = i * H;
            double exact = exact_solution(x, T1);
            rmse += pow(final_temperatures[i] - exact, 2);
        }
        rmse = sqrt(rmse / N);
        printf("RMSE: %.10f\n", rmse);
        free(initial_temperatures);
        free(final_temperatures);
    }

    free(local_temperatures);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}
