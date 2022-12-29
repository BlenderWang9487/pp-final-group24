#pragma once
#include <matrix/matrix.hpp>
#include <random>
#include <unistd.h>
#include <immintrin.h>
namespace GP{

namespace linalg{

const size_t MY_GP_CACHE_LINESIZE = 64;
matrix matmul_tile(const matrix&, const matrix&);
matrix matmul_naive(const matrix&, const matrix&);
matrix matmul_simd(const matrix&, const matrix&);
auto matmul = matmul_simd;

matrix identity(size_t n){
    matrix res{n};
    for(size_t idx = 0; idx < n; ++idx)
        res(idx, idx) = double{1};
    return res;
}


matrix randn(size_t row, size_t col = 1){
    static std::mt19937 gen(time(nullptr));
    std::normal_distribution<double> dis;
    matrix res{row, col};
    auto res_ptr = res.ptr();
    auto pad_col = res.pad_column();
    for(size_t r = 0; r < row; ++ r)
        for(size_t c = 0; c < col; ++ c)
            res_ptr[r*pad_col + c] = dis(gen);
    return res;
}


matrix diag(const matrix& m){
    auto&& [r, c] = m.shape();
    if(r == 1 or c == 1){
        int n = (r == 1 ? c : r);
        matrix res(n);
        for(size_t idx = 0; idx < n; ++ idx)
            res(idx, idx) = (r == 1 ? m(0, idx) : m(idx, 0));
        return res;
    }
    else if(r == c){
        matrix res(r, 1);
        for(size_t idx = 0; idx < r; ++ idx)
            res(idx, 0) = m(idx, idx);
        return res;
    }
    throw matrix::DimensionalityException();
}


matrix transpose(const matrix& m){
    auto&& [r, c] = m.shape();
    matrix trans{c, r};
    for(size_t i = 0;i < r;++i)
        for(size_t j = 0;j < c;++j)
            trans(j, i) = m(i, j);
    return trans;
}


matrix inv_impl(matrix& mat){
    // implement Gauss-Jordan
    auto&& [row_, col_] = mat.shape();
    if(row_ != col_){
        throw matrix::DimensionalityException();
    }
    size_t n = row_;
    size_t n_pad = mat.pad_column();
    auto inv_mat = identity(n);
    double* self_ptr = mat.ptr();
    double* inv_ptr = inv_mat.ptr();
    for(size_t iter = 0; iter < n; ++iter){
        // divide #iter row by matrix(iter, iter)
        auto self_start_iter = self_ptr + iter*n_pad;
        auto inv_start_iter = inv_ptr + iter*n_pad;
        {
            double val = mat(iter, iter);
            for(size_t c = 0; c < n; ++c)
            { self_start_iter[c] /= val; inv_start_iter[c] /= val; }
        }

        // row sub
        for(size_t r = 0; r < n; ++r){
            if(r == iter) continue;
            auto self_start_r = self_ptr + r*n_pad;
            auto inv_start_r = inv_ptr + r*n_pad;
            double ratio = mat(r, iter);
            for(size_t c = iter; c < n; ++c){
                self_start_r[c] -= self_start_iter[c] * ratio;
            }
            for(size_t c = 0; c <= iter; ++c){
                inv_start_r[c] -= inv_start_iter[c] * ratio;
            }
        }
    }
    return inv_mat;
}

matrix inv(matrix&& m){
    matrix mat = std::forward<matrix>(m);
    return inv_impl(mat);
}

matrix inv(matrix& m){
    matrix mat = m;
    return inv_impl(mat);
}


matrix matmul_tile(const matrix& a, const matrix& _b){
    auto&& [lrow, lcol] = a.shape();
    auto&& [rrow, rcol] = _b.shape();
    if(lcol != rrow){
        throw matrix::DimensionalityException();
    }
    auto b = transpose(_b);
    const size_t cache_size = MY_GP_CACHE_LINESIZE / sizeof(double);
    auto row = lrow, col = rcol, pad_inter_col = a.pad_column();
    matrix res{row, col};
    auto res_pad_col = res.pad_column();

    // things go lil bit nasty
    double* a_ptr = a.ptr();
    double* b_ptr = b.ptr();
    double* res_ptr = res.ptr();
    for(size_t r = 0; r < row; r += cache_size){
        size_t r_max = std::min(r + cache_size, row);
        for(size_t c = 0; c < col; c += cache_size){
            size_t c_max = std::min(c + cache_size, col);
            for(size_t k = 0; k < lcol; k += cache_size){
                size_t k_max = std::min(k + cache_size, lcol);
                for(size_t r_tile = r; r_tile < r_max; ++ r_tile){
                    auto a_start  = a_ptr + r_tile*pad_inter_col;
                    for(size_t c_tile = c; c_tile < c_max; ++ c_tile){
                        auto b_start = b_ptr + c_tile*pad_inter_col;
                        double sum{};
                        for(size_t k_tile = k; k_tile < k_max; ++ k_tile)
                            sum += a_start[k_tile] * b_start[k_tile];
                        res_ptr[r_tile*res_pad_col + c_tile] += sum;
                    }
                }
            }
        }
    }
    return res;
}
matrix matmul_naive(const matrix& a, const matrix& _b){
    auto&& [lrow, lcol] = a.shape();
    auto&& [rrow, rcol] = _b.shape();
    if(lcol != rrow){
        throw matrix::DimensionalityException();
    }
    auto b = transpose(_b);
    const size_t cache_size = MY_GP_CACHE_LINESIZE / sizeof(double);
    auto row = lrow, col = rcol, pad_inter_col = a.pad_column(), pad_b = b.pad_column();
    matrix res{row, col};
    auto res_pad_col = res.pad_column();

    // things go lil bit nasty
    double* a_ptr = a.ptr();
    double* b_ptr = b.ptr();
    double* res_ptr = res.ptr();
    for(size_t r = 0; r < row; ++r){
        for(size_t c = 0; c < col; ++c){
            double sum{};
            for(size_t k = 0; k < lcol; ++k)
                sum += a_ptr[r*pad_inter_col + k] * b_ptr[c*pad_inter_col + k];
            res_ptr[r*res_pad_col + c] = sum;
        }
    }
    return res;
}

// matrix matmul_simd(const matrix& a, const matrix& _b){
matrix matmul_simd(const matrix& a, const matrix& _b){
    auto&& [lrow, lcol] = a.shape();
    auto&& [rrow, rcol] = _b.shape();
    if(lcol != rrow){
        throw matrix::DimensionalityException();
    }
    auto b = transpose(_b);
    const size_t cache_size = MY_GP_CACHE_LINESIZE / sizeof(double);
    auto row = lrow, col = rcol, pad_inter_col = a.pad_column();
    matrix res{row, col};
    auto res_pad_col = res.pad_column();

    // things go lil bit nasty
    double* a_ptr = a.ptr();
    double* b_ptr = b.ptr();
    double* res_ptr = res.ptr();
    __m128d _upper;
    __m128d _lower;
    double sum[2];
    for(size_t r = 0; r < row; r += cache_size){
        size_t r_max = std::min(r + cache_size, row);
        for(size_t c = 0; c < col; c += cache_size){
            size_t c_max = std::min(c + cache_size, col);
            for(size_t k = 0; k < lcol; k += cache_size){
                size_t k_max = std::min(k + cache_size, lcol);
                size_t n = k_max - k;
                for(size_t r_tile = r; r_tile < r_max; ++ r_tile){
                    auto a_start  = a_ptr + r_tile*pad_inter_col;
                    for(size_t c_tile = c; c_tile < c_max; ++ c_tile){
                        auto b_start = b_ptr + c_tile*pad_inter_col;
                        if(n == cache_size){
                            auto sum1 = _mm256_hadd_pd(
                                _mm256_mul_pd(
                                    _mm256_load_pd(a_start), _mm256_load_pd(b_start)
                                ),
                                _mm256_mul_pd(
                                    _mm256_load_pd(a_start+4), _mm256_load_pd(b_start+4)
                                )
                            );
                            sum1 = _mm256_hadd_pd(sum1, sum1);
                            _upper = _mm256_extractf128_pd(sum1, 0);
                            _lower = _mm256_extractf128_pd(sum1, 1);
                            _mm_store1_pd(sum, _upper);
                            _mm_store1_pd(sum+1, _lower);
                        }else{
                            sum[0] = 0., sum[1] = 0.;
                            for(size_t k_tile = k; k_tile < k_max; ++ k_tile)
                                *sum += a_start[k_tile] * b_start[k_tile];
                        }
                        res_ptr[r_tile*res_pad_col + c_tile] += sum[0] + sum[1];
                    }
                }
            }
        }
    }
    return res;
}
 
matrix operator^(const matrix& lhs, const matrix& rhs){
    return matmul(lhs, rhs);
}

 
matrix& operator^=(matrix& lhs, const matrix& rhs){
    lhs = matmul(lhs, rhs);
    return lhs;
}
 
matrix operator~(matrix&& m){
    matrix mat = std::forward<matrix>(m);
    return inv_impl(mat);
}
 
matrix operator~(matrix& m){
    matrix mat = m;
    return inv_impl(mat);
}

}
}