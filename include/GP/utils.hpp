#pragma once
#include <matrix/matrix.hpp>

namespace GP{

namespace utils{

double sum(const matrix& m){
    double* m_ptr = m.ptr();
    auto && [row, col] = m.shape();
    auto n = row*col;
    double s{};
    for(size_t idx = 0; idx < n; ++idx)
        s += m_ptr[idx];
    return s;
}

double mean(const matrix& m){
    double s{sum(m)};
    auto && [row, col] = m.shape();
    auto n = m.size();
    return s / n;
}

}

}