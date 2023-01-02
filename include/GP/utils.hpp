#pragma once
#include <matrix/matrix.hpp>
#include <chrono>

namespace GP{

namespace utils{

double sum(const matrix& m){
    double* m_ptr = m.ptr();
    auto && [row, col] = m.shape();
    auto pad_col = m.pad_column();
    double s{};
    for(size_t r = 0; r < row; ++ r)
        for(size_t c = 0; c < col; ++ c)
            s += m_ptr[r*pad_col + c];
    return s;
}

double mean(const matrix& m){
    return sum(m) / m.size();
}

auto print_time_spent(std::chrono::high_resolution_clock::time_point start_time){
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = end_time - start_time;
    std::cout << "Time spent: " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
        << " (ms)"
        << std::endl;
    return end_time;
}

}

}