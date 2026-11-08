// omp_check.c - sanity check that OpenMP works and how many threads we can use.
#include <stdio.h>
#include <omp.h>

int main(void) {
    printf("OpenMP version macro : %d\n", _OPENMP);
    printf("Max threads (default): %d\n", omp_get_max_threads());
    printf("Logical processors   : %d\n", omp_get_num_procs());

    int counts[] = {1, 2, 4, 8, 16};
    for (int i = 0; i < 5; i++) {
        omp_set_num_threads(counts[i]);
        int got = 0;
        #pragma omp parallel
        {
            #pragma omp single
            got = omp_get_num_threads();
        }
        printf("requested %2d threads -> got %2d\n", counts[i], got);
    }

    // tiny parallel loop: sum 0..N-1, result must be N*(N-1)/2
    const long N = 100000000L;
    double t0 = omp_get_wtime();
    long long sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < N; i++) sum += i;
    double t1 = omp_get_wtime();
    printf("sum = %lld (expected %lld), time = %.3f s\n",
           sum, (long long)N * (N - 1) / 2, t1 - t0);
    return 0;
}
