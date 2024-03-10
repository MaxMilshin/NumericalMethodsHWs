#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>

#include "preset.h"
#include "parallel.h"
#include "consecutive.h"

enum SOLUTION {
    PARALLEL,
    CONSECUTIVE
};

void reset_values(std::vector<std::vector<double>>& u) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (int i = 0; i < N + 2; i++) {
        for (int j = 0; j < N + 2; j++) {
            double x = static_cast<double>(i) / (N + 1);
            double y = static_cast<double>(j) / (N + 1);
            if (y == 0) {
                u[i][j] = 100 - 200 * x;
            }
            else if (x == 0) {
                u[i][j] = 100 - 200 * y;
            }
            else if (y == 1) {
                u[i][j] = -100 + 200 * x;
            }
            else if (x == 1) {
                u[i][j] = -100 + 200 * y;
            }
            else {
                u[i][j] = dis(gen);
            }
        }
    }
}

void make_run(const SOLUTION& solution, std::vector<std::vector<double>>& u, const std::vector<std::vector<double>>& f, int threads_number = 8) {
    reset_values(u);

    auto start = std::chrono::high_resolution_clock::now();
    int iterations = (solution == PARALLEL) ? solve_parallel(u, f, threads_number) : solve_consecutive(u, f);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << ((solution == PARALLEL) ? "PARALLEL": "CONSECUTIVE") << " SOLUTION" << ((solution == PARALLEL) ? " FOR " + std::to_string(threads_number) + " THREADS" : "") << std::endl;
    std::cout << "Time taken by: " << static_cast<double>(duration.count()) / 1e6 << " seconds" << std::endl;
    std::cout << "Iterations count: " << iterations << std::endl;
    std::cout << std::endl;
}

int main() {
    std::vector<std::vector<double>> f(N + 2, std::vector<double> (N + 2));

    std::vector<std::vector<double>> u(N + 2, std::vector<double> (N + 2));


    make_run(CONSECUTIVE, u, f);
    // std::vector<int> threads_number_options = {1, 2, 4, 6, 8, 10, 12, 14, 15, 16};
    std::vector<int> threads_number_options = {1, 14, 16, 18, 20};
    // std::vector<int> threads_number_options = {1, 16};
    for (int threads_number : threads_number_options) {
        make_run(PARALLEL, u, f, threads_number);
    }
    std::cout << std::endl << std::endl;

    // make_run(CONSECUTIVE, u, f);
    // std::vector<int> block_sizes = {60, 64, 68, 80, 96, 100, 120, 128, 144, 160, 192, 224};
    // std::vector<int> threads_number_options = {4, 6, 8, 12, 14, 15, 16};
    // for (int block_size : block_sizes) {
    //     std::cout << "BLOCK SIZE: " << block_size << std::endl;
    //     // make_run(PARALLEL, u, f, 1, block_size);
    //     for (int threads_number : threads_number_options) {
    //         make_run(PARALLEL, u, f, threads_number, block_size);
    //     }
    //     std::cout << std::endl << std::endl;
    // }
}
