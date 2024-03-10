#include <iostream>
#include <vector>
#include <cmath>

#include "preset.h"

int solve_consecutive(std::vector<std::vector<double>>& u, const std::vector<std::vector<double>>& f) {
    int iterations = 0;
    double dmax = 0;
    do {
        dmax = 0;
        iterations++;
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                double temp = u[i][j];
                u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h * f[i][j]);
                double diff_value = fabs(u[i][j] - temp);
                if (diff_value > dmax) {
                    dmax = diff_value;
                }
            }
        }
    } while (dmax > EPS);
    return iterations;
}
//
// Created by bethi on 29.02.2024.
//
