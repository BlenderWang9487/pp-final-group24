#pragma once
#include <matrix/matrix.hpp>
#include <linalg/linalg.hpp>
#include <cmath>
#include <numeric>
#include <vector>
#include <immintrin.h>

namespace GP{

class GPRegression{
private:
    matrix C_inv_, X_, Y_;
    double gamma_;
    double beta_;
public:
    matrix rbf_kernel_naive_(const matrix&, const matrix&);
    matrix rbf_kernel_simd_(const matrix&, const matrix&);
    GPRegression(double g = 0.1, double b = double{1}):
        C_inv_{}, X_{}, Y_{}, gamma_{g}, beta_{b} {}
    void fit(const matrix& X, const matrix& Y){
        using namespace GP::linalg;
        auto&& [xr, xc] = X.shape();
        auto&& [yr, yc] = Y.shape();
        if(xr != yr){
            throw matrix::DimensionalityException();
        }
        auto start_time = std::chrono::high_resolution_clock::now();
        X_ = X; Y_ = Y;
        std::cout << "[X_, Y_ copy] ";
        start_time = GP::utils::print_time_spent(start_time);
        auto K = rbf_kernel_simd_(X, X);
        std::cout << "[Kernel] ";
        start_time = GP::utils::print_time_spent(start_time);
        C_inv_ = ~(K + identity(xr) * beta_);
        std::cout << "[Inverse] ";
        start_time = GP::utils::print_time_spent(start_time);
    }
    auto predict(const matrix& X_test){
        using namespace GP::linalg;
        auto&& [xr, xc] = X_.shape();
        auto&& [xtest_r, xtest_c] = X_test.shape();
        if(xc != xtest_c){
            throw matrix::DimensionalityException();
        }
        auto k = rbf_kernel_simd_(X_, X_test);
        auto ktCinv = transpose(k) ^ C_inv_;
        return std::pair<matrix, matrix>{
            ktCinv ^ Y_,
            rbf_kernel_simd_(X_test, X_test) + identity(xtest_r) * beta_ - (ktCinv ^ k)
        };
    }
};

matrix GPRegression::rbf_kernel_naive_(const matrix& X1, const matrix& X2){
    using namespace GP::linalg;
    auto&& [n1, feats1] = X1.shape();
    auto&& [n2, feats2] = X2.shape();
    if(feats1 != feats2){
        throw matrix::DimensionalityException();
    }
    matrix kernel(n1, n2);
    for(size_t r = 0; r < n1; ++r){
        for(size_t c = 0; c < n2; ++c){
            std::vector<double> vec_dif(feats1);
            for(size_t k = 0; k < feats1; ++k)
                vec_dif[k] = X1(r, k) - X2(c, k);
            double dot_product = std::inner_product(
                vec_dif.begin(), vec_dif.end(), vec_dif.begin(), double{});
            kernel(r, c) = std::exp(-gamma_*dot_product);
        }
    }
    return kernel;
}

matrix GPRegression::rbf_kernel_simd_(const matrix& X1, const matrix& X2){
    using namespace GP::linalg;
    auto&& [n1, feats1] = X1.shape();
    auto&& [n2, feats2] = X2.shape();
    if(feats1 != feats2){
        throw matrix::DimensionalityException();
    }
    size_t pad_feature = X1.pad_column();
    constexpr size_t cache_size = 4;
    constexpr auto shuffle = _MM_SHUFFLE(3,1,2,0);
    auto x1_ptr = X1.ptr();
    auto x2_ptr = X2.ptr();
    matrix kernel(n1, n2);
    auto k_ptr = kernel.ptr();
    size_t kernel_pad = kernel.pad_column();
    for(size_t r = 0; r < n1; ++r){
        auto k_start = k_ptr + r*kernel_pad;
        for(size_t c = 0; c < n2; c += cache_size){
            size_t c_max = std::min(c + cache_size, n2);
            size_t n_c = c_max - c;
            if(n_c == cache_size){
                auto c_sum_vec = _mm256_setzero_pd();
                for(size_t k = 0; k < feats1; k += cache_size){
                    auto x2_start = x2_ptr + c*pad_feature + k;
                    auto x1_vec = _mm256_load_pd(x1_ptr + r*pad_feature + k);

                    // xdif = xi - xj
                    auto dif1 = _mm256_sub_pd(x1_vec, _mm256_load_pd(x2_start));
                    auto dif2 = _mm256_sub_pd(x1_vec, _mm256_load_pd(x2_start + pad_feature));
                    auto dif3 = _mm256_sub_pd(x1_vec, _mm256_load_pd(x2_start + pad_feature*2));
                    auto dif4 = _mm256_sub_pd(x1_vec, _mm256_load_pd(x2_start + pad_feature*3));
                    
                    // xdif = xdif^2
                    dif1 = _mm256_mul_pd(dif1, dif1);
                    dif2 = _mm256_mul_pd(dif2, dif2);
                    dif3 = _mm256_mul_pd(dif3, dif3);
                    dif4 = _mm256_mul_pd(dif4, dif4);

                    // x_l2 = sum of xdif^2
                    c_sum_vec = _mm256_add_pd(c_sum_vec, _mm256_permute4x64_pd(_mm256_hadd_pd(
                        _mm256_permute4x64_pd(_mm256_hadd_pd(dif1, dif2), shuffle),
                        _mm256_permute4x64_pd(_mm256_hadd_pd(dif3, dif4), shuffle)
                    ), shuffle));
                }
                // kernel = -gamma * x_l2
                _mm256_store_pd(k_start + c, _mm256_mul_pd(_mm256_set1_pd(-gamma_), c_sum_vec));
            }
            else{
                for(size_t c_tile = c; c_tile < c_max; ++c_tile){
                    std::vector<double> vec_dif(feats1);
                    for(size_t k = 0; k < feats1; ++k)
                        vec_dif[k] = x1_ptr[r*pad_feature + k] - x2_ptr[c_tile*pad_feature + k];
                    k_start[c_tile] = -gamma_*std::inner_product(vec_dif.begin(), vec_dif.end(), vec_dif.begin(), double{});
                }
            }
        }
    }
    for(size_t r = 0; r < n1; ++r)
        for(size_t c = 0; c < n2; ++c)
            k_ptr[r*kernel_pad + c] = std::exp(k_ptr[r*kernel_pad + c]);
    return kernel;
}

}