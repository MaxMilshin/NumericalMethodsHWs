//
// Created by bethi on 29.02.2024.
//

#ifndef PRESET_H
#define PRESET_H

constexpr int N = 3000;
constexpr int BLOCK_SIZE = 160;
constexpr int BLOCKS_NUMBER = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
constexpr double EPS = 1e-1;
constexpr unsigned int seed = 12345;

constexpr double h = static_cast<double>(1) / (N + 1);

#endif //PRESET_H
