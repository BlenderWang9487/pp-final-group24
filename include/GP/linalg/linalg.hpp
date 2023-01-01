#pragma once
#include <matrix/matrix.hpp>
#include <random>
#include <unistd.h>
#include <immintrin.h>
#include <omp.h>
namespace GP{

namespace linalg{

const size_t MY_GP_CACHE_LINESIZE = 256;
matrix matmul_tile(const matrix&, const matrix&);
matrix matmul_naive(const matrix&, const matrix&);
matrix matmul_simd(const matrix&, const matrix&);
matrix matmul_simd2(const matrix&, const matrix&);
matrix inv_impl_naive(matrix&);
matrix inv_impl_simd(matrix&);
auto matmul = matmul_simd2;
auto inv_impl = inv_impl_simd;
// auto matmul = matmul_tile;
// auto inv_impl = inv_impl_naive;

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


matrix inv_impl_naive(matrix& mat){
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
matrix inv_impl_simd(matrix& mat){
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
        double val = mat(iter, iter);
        for(size_t c = 0; c < n; ++c){
            self_start_iter[c] /= val; inv_start_iter[c] /= val;
        }

        // row sub
        __m256d ratio_vec;
        for(size_t r = 0; r < n; ++r){
            if(r == iter) continue;
            auto self_start_r = self_ptr + r*n_pad;
            auto inv_start_r = inv_ptr + r*n_pad;
            double ratio = mat(r, iter);
            ratio_vec = _mm256_set1_pd(ratio);
            
            size_t iter_simd_max = (iter+1)/matrix::simd_len*matrix::simd_len;
            for(size_t c = 0; c < iter_simd_max; c += matrix::simd_len){
                _mm256_store_pd(inv_start_r+c, _mm256_sub_pd(
                    _mm256_load_pd(inv_start_r+c),
                    _mm256_mul_pd(_mm256_load_pd(inv_start_iter+c), ratio_vec)));
            }
            for(size_t c = iter_simd_max; c <= iter; ++ c){
                inv_start_r[c] -= inv_start_iter[c] * ratio;
            }
            iter_simd_max = n/matrix::simd_len*matrix::simd_len;
            for(size_t c = iter/matrix::simd_len*matrix::simd_len; c < iter_simd_max; c += matrix::simd_len){
                _mm256_store_pd(self_start_r+c, _mm256_sub_pd(
                    _mm256_load_pd(self_start_r+c),
                    _mm256_mul_pd(_mm256_load_pd(self_start_iter+c), ratio_vec)));
            }
            for(size_t c = iter_simd_max; c < n; ++ c){
                self_start_r[c] -= self_start_iter[c] * ratio;
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
inline
double hsum_double_avx(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
            vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}
inline
__m256d double16_hadd(double* a_start, double* b_start){
    return _mm256_hadd_pd(
        _mm256_hadd_pd(
            _mm256_hadd_pd(
                _mm256_mul_pd(_mm256_load_pd(a_start), _mm256_load_pd(b_start)),
                _mm256_mul_pd(_mm256_load_pd(a_start+4), _mm256_load_pd(b_start+4))
            ),
            _mm256_hadd_pd(
                _mm256_mul_pd(_mm256_load_pd(a_start+8), _mm256_load_pd(b_start+8)),
                _mm256_mul_pd(_mm256_load_pd(a_start+12), _mm256_load_pd(b_start+12))
            )),
        _mm256_hadd_pd(
            _mm256_hadd_pd(
                _mm256_mul_pd(_mm256_load_pd(a_start+16), _mm256_load_pd(b_start+16)),
                _mm256_mul_pd(_mm256_load_pd(a_start+20), _mm256_load_pd(b_start+20))
            ),
            _mm256_hadd_pd(
                _mm256_mul_pd(_mm256_load_pd(a_start+24), _mm256_load_pd(b_start+24)),
                _mm256_mul_pd(_mm256_load_pd(a_start+28), _mm256_load_pd(b_start+28))
            )));
}
inline
__m256d double4row_hadd(double* a_start, double* b_start, size_t pad_col_ab, const int imm8){
    auto a_vec = _mm256_load_pd(a_start);
    return _mm256_permute4x64_pd(_mm256_hadd_pd(
            _mm256_permute4x64_pd(_mm256_hadd_pd(
                _mm256_mul_pd(a_vec, _mm256_load_pd(b_start)),
                _mm256_mul_pd(a_vec, _mm256_load_pd(b_start + pad_col_ab))
            ), imm8),
            _mm256_permute4x64_pd(_mm256_hadd_pd(
                _mm256_mul_pd(a_vec, _mm256_load_pd(b_start + pad_col_ab*2)),
                _mm256_mul_pd(a_vec, _mm256_load_pd(b_start + pad_col_ab*3))
            ), imm8)
        ), imm8);
}
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
    for(size_t r = 0; r < row; r += cache_size){
        size_t r_max = std::min(r + cache_size, row);
        for(size_t c = 0; c < col; c += cache_size){
            size_t c_max = std::min(c + cache_size, col);
            for(size_t k = 0; k < lcol; k += cache_size){
                size_t k_max = std::min(k + cache_size, lcol);
                size_t n = k_max - k;
                for(size_t r_tile = r; r_tile < r_max; ++ r_tile){
                    auto a_start = a_ptr + r_tile*pad_inter_col + k;
                    for(size_t c_tile = c; c_tile < c_max; ++ c_tile){
                        double sum = 0.;
                        auto b_start = b_ptr + c_tile*pad_inter_col + k;
                        if(n == cache_size){
                            sum = hsum_double_avx(double16_hadd(a_start, b_start));
                        }else{
                            for(size_t k_tile = 0; k_tile < n; ++ k_tile)
                                sum += a_start[k_tile] * b_start[k_tile];
                        }
                        res_ptr[r_tile*res_pad_col + c_tile] += sum;
                    }
                }
            }
        }
    }
    return res;
}
matrix matmul_simd2(const matrix& a, const matrix& _b){
    auto&& [lrow, lcol] = a.shape();
    auto&& [rrow, rcol] = _b.shape();
    if(lcol != rrow){
        throw matrix::DimensionalityException();
    }
    auto b = transpose(_b);
    const size_t cache_size = 4;
    // const size_t cache_size = MY_GP_CACHE_LINESIZE / sizeof(double);
    auto row = lrow, col = rcol, pad_inter_col = a.pad_column();
    matrix res{row, col};
    auto res_pad_col = res.pad_column();
    auto shuffle = _MM_SHUFFLE(3,1,2,0);

    // things go lil bit nasty
    double* a_ptr = a.ptr();
    double* b_ptr = b.ptr();
    double* res_ptr = res.ptr();
    for(size_t r = 0; r < row; r += cache_size){
        size_t r_max = std::min(r + cache_size, row);
        for(size_t c = 0; c < col; c += cache_size){
            size_t c_max = std::min(c + cache_size, col);
            size_t n_c = c_max - c;
            for(size_t k = 0; k < lcol; k += cache_size){
                size_t k_max = std::min(k + cache_size, lcol);
                size_t n = k_max - k;
                for(size_t r_tile = r; r_tile < r_max; ++ r_tile){
                    auto a_start = a_ptr + r_tile*pad_inter_col + k;
                    auto res_start = res_ptr + r_tile*res_pad_col;
                    if(n_c == cache_size && n == cache_size){
                        _mm256_store_pd(
                            res_start + c, _mm256_add_pd(
                                _mm256_load_pd(res_start + c),
                                double4row_hadd(a_start, b_ptr + c*pad_inter_col + k, pad_inter_col, shuffle)
                            ));
                    }
                    else{
                        for(size_t c_tile = c; c_tile < c_max; ++ c_tile){
                            auto b_start = b_ptr + c_tile*pad_inter_col + k;
                            double sum = 0.;
                            for(size_t k_tile = 0; k_tile < n; ++ k_tile)
                                sum += a_start[k_tile] * b_start[k_tile];
                            res_start[c_tile] += sum;
                        }
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