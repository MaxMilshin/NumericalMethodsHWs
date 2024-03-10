// #include <iostream>
#include <omp.h>
#include <vector>
#include <random>
#include <iostream>

#include "preset.h"


int solve_parallel(std::vector<std::vector<double>>& u, const std::vector<std::vector<double>>& f, int threads_number) {
    omp_lock_t dmax_lock;
    omp_init_lock(&dmax_lock);

    std::vector<double> diff(N + 1, 0.0);

    // int BLOCK_SIZE = block_size;
    // int BLOCKS_NUMBER = (N + block_size - 1) / block_size;

    int iterations = 0;
    double dmax = 0;
    do {
        // if (iterations == 10) {
        //     break;
        // }
        iterations++;
        // std::cout << "brand new iteration" << std::endl;
        if (iterations > 1) {
            std::fill(diff.begin(), diff.end(), 0.0);
        }
        for (int block_diag_num = 0; block_diag_num < BLOCKS_NUMBER; block_diag_num++) {
            int i, j, ii, jj;
            double temp, diff_value;

            #pragma omp parallel for shared(u, f, diff, block_diag_num, BLOCK_SIZE, N) private(i, j, ii, jj, temp, diff_value) num_threads(threads_number)
            for (i = 0; i <= block_diag_num; i++) {
                // int threadID = omp_get_thread_num();
                // std::cout << "Thread: " << threadID << std::endl;
                j = block_diag_num - i;
                // proccessing big block (i,j)
                for (ii = i * BLOCK_SIZE + 1; ii <= std::min(N, (i + 1) * BLOCK_SIZE); ii++) {
                    for (jj = j * BLOCK_SIZE + 1; jj <= std::min(N, (j + 1) * BLOCK_SIZE); jj++) {
                        // a[ii][jj] = std::to_string(i) + std::to_string(j);
                        temp = u[ii][jj];
                        u[ii][jj] = 0.25 * (u[ii - 1][jj] + u[ii + 1][jj] + u[ii][jj - 1] + u[ii][jj + 1] - h * h * f[ii][jj]);
                        diff_value = fabs(temp - u[ii][jj]);
                        if (diff_value > diff[ii]) {
                            diff[ii] = diff_value;
                        }
                    }
                }
            } // end of parallel area
        }
        // wave fadding
        for (int block_diag_num = BLOCKS_NUMBER; block_diag_num <= 2 * BLOCKS_NUMBER - 2; block_diag_num++) {
            int i, j, ii, jj;
            double temp, diff_value;
            #pragma omp parallel for shared(u, f, diff, block_diag_num, BLOCK_SIZE, BLOCKS_NUMBER, N) private(i, j, ii, jj, temp, diff_value) default(none) num_threads(threads_number)
            for (i = BLOCKS_NUMBER - 1; i >= block_diag_num - (BLOCKS_NUMBER - 1); i--) {
                j = block_diag_num - i;
                // proccessing big block (i,j)
                for (ii = i * BLOCK_SIZE + 1; ii <= std::min(N, (i + 1) * BLOCK_SIZE); ii++) {
                    for (jj = j * BLOCK_SIZE + 1; jj <= std::min(N, (j + 1) * BLOCK_SIZE); jj++) {
                        // a[ii][jj] = std::to_string(i) + std::to_string(j);
                        temp = u[ii][jj];
                        u[ii][jj] = 0.25 * (u[ii - 1][jj] + u[ii + 1][jj] + u[ii][jj - 1] + u[ii][jj + 1] - h * h * f[ii][jj]);
                        diff_value = fabs(temp - u[ii][jj]);
                        if (diff_value > diff[ii]) {
                            diff[ii] = diff_value;
                        }
                    }
                }
            }
        }
        int chunk = 120;
        int i, j;
        double d;
        dmax = 0;
        #pragma omp parallel for shared(diff, dmax, N, chunk, dmax_lock) private(i, j, d) default(none) num_threads(threads_number)
        for (i = 1; i <= N; i += chunk) {
            d = 0;
            for (j = i; j < std::min(i + chunk, N + 1); j++)
                if (d < diff[j]) {
                    d = diff[j];
                }
            omp_set_lock(&dmax_lock);
            if (dmax < d) {
                dmax = d;
            }
            omp_unset_lock(&dmax_lock);
        } // end of parallel area
    } while (dmax > EPS);
    return iterations;
}