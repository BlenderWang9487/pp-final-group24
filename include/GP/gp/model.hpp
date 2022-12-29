#pragma once
#include <matrix/matrix.hpp>
#include <linalg/linalg.hpp>
#include <cmath>
#include <numeric>
#include <vector>

namespace GP{

class GPRegression{
private:
    matrix C_inv_, X_, Y_;
    double gamma_;
    double beta_;
    matrix rbf_kernel_(const matrix&, const matrix&);
public:
    GPRegression(double g = 0.1, double b = double{1}):
        C_inv_{}, X_{}, Y_{}, gamma_{g}, beta_{b} {}
    void fit(const matrix& X, const matrix& Y){
        using namespace GP::linalg;
        auto&& [xr, xc] = X.shape();
        auto&& [yr, yc] = Y.shape();
        if(xr != yr){
            throw matrix::DimensionalityException();
        }
        X_ = X; Y_ = Y;
        C_inv_ = ~(rbf_kernel_(X, X) + identity(xr) * beta_);
    }
    auto predict(const matrix& X_test){
        using namespace GP::linalg;
        auto&& [xr, xc] = X_.shape();
        auto&& [xtest_r, xtest_c] = X_test.shape();
        if(xc != xtest_c){
            throw matrix::DimensionalityException();
        }
        auto k = rbf_kernel_(X_, X_test);
        auto ktCinv = transpose(k) ^ C_inv_;
        return std::pair<matrix, matrix>{
            ktCinv ^ Y_,
            rbf_kernel_(X_test, X_test) + identity(xtest_r) * beta_ - (ktCinv ^ k)
        };
    }
};

matrix GPRegression::rbf_kernel_(const matrix& X1, const matrix& X2){
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

}