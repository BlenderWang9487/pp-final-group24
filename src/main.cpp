#include <matrix/matrix.hpp>
#include <linalg/linalg.hpp>
#include <utils.hpp>
#include <gp/model.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

auto statistic(GP::matrix& m){
    auto&& [row, col] = m.shape();

    // mean
    GP::matrix mu{1, col};
    for(size_t r = 0; r < row; ++ r)
        for(size_t c = 0; c < col; ++ c)
            mu(0, c) += m(r, c);
    mu *= double{1./row};
    
    // var
    GP::matrix stdv{1, col};
    for(size_t r = 0; r < row; ++ r)
        for(size_t c = 0; c < col; ++ c){
            auto dif = (m(r, c) - mu(0, c));
            stdv(0, c) += dif*dif;
        }
    stdv *= double{1./row};
    for(size_t c = 0; c < col; ++ c){
        stdv(0, c) = sqrt(stdv(0, c));
    }
    return std::pair<GP::matrix, GP::matrix>{mu, stdv};
}


void preprocess(GP::matrix& m, const GP::matrix& mu, const GP::matrix& stdv){
    auto&& [row, col] = m.shape();
    for(size_t r = 0; r < row; ++ r)
        for(size_t c = 0; c < col; ++ c)
            m(r, c) = stdv(0, c) != 0 ? ((m(r, c) - mu(0, c)) / stdv(0, c)) : (m(r, c) - mu(0, c));
}

void train(){
    using namespace GP::linalg;
    GP::matrix X, X_test;
    GP::matrix Y, Y_test;
    std::cin >> X >> Y;
    std::cin >> X_test >> Y_test;

    auto&& [col_mu_X, col_stdv_X] = statistic(X);
    preprocess(X, col_mu_X, col_stdv_X);
    preprocess(X_test, col_mu_X, col_stdv_X);

    std::cout << "train data size:" << X.shape().first << '\n';
    std::cout << "test data size:" << X_test.shape().first << '\n';
    std::cout << "preprocess mu:\n" << col_mu_X;
    std::cout << "preprocess std:\n" << col_stdv_X;
    int g = 1, b = 1;
    for(g = 1; g < 10001; g *= 10)
    for(b = 1; b < 10001; b *= 10) // grid search
    {
        std::cout << std::string(70, '-') << '\n';
        std::cout << "para: gamma = " << g*0.00003 <<", beta = " << b*0.01 << std::endl;
        GP::GPRegression model{g*0.000003, b*0.01};

        auto start = std::chrono::high_resolution_clock::now();
        model.fit(X, Y);
        std::cout << "[Fit] ";
        start = GP::utils::print_time_spent(start);
        auto&& [mu, var] = model.predict(X_test);
        std::cout << "[Predict] ";
        start = GP::utils::print_time_spent(start);

        // mse 
        auto diff = (mu - Y_test);
        std::cout << "MSE: " << (transpose(diff) ^ diff) * (1./diff.size())<<"\n";

        auto compare = GP::matrix{Y_test.size(), 2};
        for(size_t idx = 0; idx < Y_test.size(); ++idx){
            compare(idx, 0) = mu(idx, 0);
            compare(idx, 1) = Y_test(idx, 0);
        }
        // std::cout << "Compare:\n" << compare;
        // std::cout << "X:\n" << X;
        // std::cout << "X_test:\n" << X_test;

    }
}

void linalg_benchmark(){
    using namespace GP::linalg;
    std::vector<GP::matrix> mats;
    size_t size_list[] = {100, 300, 1000};
    for(auto s : size_list)
        mats.emplace_back(randn(s, s));
    int repeat = 5;

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "[matmul] Repeat " << repeat <<" times \n";
    for(auto& mat : mats){
        std::cout << "Matirx "<< mat.shape(0) << "x" << mat.shape(1) << " - ";
        for(int i = 0; i < repeat; ++i)
            mat ^= mat;
        start = GP::utils::print_time_spent(start);
    }
    std::cout << "[inv] Repeat " << repeat <<" times \n";
    for(auto& mat : mats){
        std::cout << "Matirx "<< mat.shape(0) << "x" << mat.shape(1) << " - ";
        for(int i = 0; i < repeat; ++i)
            mat = ~mat;
        start = GP::utils::print_time_spent(start);
    }
}

void valid_matmul(){
    using namespace GP::linalg;
    auto mat = randn(10, 10);
    auto idt = identity(10);
    auto mat_inv = ~mat;
    std::cout << "mat @ inv_mat\n" << (mat ^ mat_inv);
    std::cout << "idt * 2\n" << (idt * 2.);
    std::cout << "idt + 2\n" << (idt + 2.);
    std::cout << "(idt - 2.) * 4 - 2\n" << (idt - 2.) * 4 - 2;
    std::cout << "pad col " << idt.pad_column() << ", " << mat.pad_column() << '\n';
}

void matmul_benchmark(){
    using namespace GP::linalg;
    std::vector<GP::matrix> mats;
    size_t size_list[] = {1000};
    for(auto s : size_list)
        mats.emplace_back(randn(s, s));
    int repeat = 5;
    auto perform_bm = [&repeat](const GP::matrix& mat, auto impl){
        std::cout << "Matirx "<< mat.shape(0) << "x" << mat.shape(1) << " - ";
        auto cpy = mat;
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < repeat; ++i)
            cpy = impl(cpy, cpy);
        start = GP::utils::print_time_spent(start);
    };
    auto impl_list = {
        std::pair{matmul_simd, "matmul_simd"},
        std::pair{matmul_naive, "matmul_naive"},
        std::pair{matmul_tile, "matmul_tile"},
        std::pair{matmul_simd2, "matmul_simd2"}
    };
    for(auto&& [impl, impl_name] : impl_list){
        std::cout << "[" << impl_name << "] Repeat " << repeat <<" times \n";
        for(auto& mat : mats){
            perform_bm(mat, impl);
        }
    }
}

void inv_benchmark(){
    using namespace GP::linalg;
    std::vector<GP::matrix> mats;
    size_t size_list[] = {128, 512, 1024, 1200};
    for(auto s : size_list)
        mats.emplace_back(randn(s, s));
    int repeat = 5;
    auto perform_bm = [&repeat](const GP::matrix& mat, auto impl){
        std::cout << "Matirx "<< mat.shape(0) << "x" << mat.shape(1) << " - ";
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < repeat; ++i){
            auto cpy = mat;
            cpy = impl(cpy);
        }
        start = GP::utils::print_time_spent(start);
    };
    auto impl_list = {
        std::pair{inv_impl, "inv_impl"},
        std::pair{inv_impl_simd, "inv_impl_simd"},
    };
    for(auto&& [impl, impl_name] : impl_list){
        std::cout << "[" << impl_name << "] Repeat " << repeat <<" times \n";
        for(auto& mat : mats){
            perform_bm(mat, impl);
        }
    }
    std::cout << "inv Validation:\n";
    for(auto&& [impl, impl_name] : impl_list){
        std::cout << impl_name << ": ||mat ^ inv_mat - identity|| = \n";
        auto mat = randn(100, 100);
        auto tmp = mat;
        auto mat_inv = impl(tmp);
        auto dif = (mat ^ mat_inv) - identity(100);
        dif = dif * dif;
        std::cout << GP::utils::mean(dif) << '\n';
    }
}
void kernel_benchmark(){
    using namespace GP::linalg;
    std::vector<GP::matrix> mats;
    size_t size_list[] = {128, 512, 1024, 1200, 2000};
    int repeat = 5;
    size_t feat = 123;
    auto perform_bm = [&repeat, &feat](const size_t size_x, GP::GPRegression& model){
        std::cout << "[serial] Repeat " << repeat <<" times \n";
        std::cout << "Matirx "<< size_x << "x" << feat << " - ";
        auto mat = randn(size_x, feat);
        GP::matrix res, res_simd;
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < repeat; ++i){
            res = model.rbf_kernel_naive_(mat, mat);
        }
        start = GP::utils::print_time_spent(start);
        std::cout << "[simd] Repeat " << repeat <<" times \n";
        std::cout << "Matirx "<< mat.shape(0) << "x" << mat.shape(1) << " - ";
        for(int i = 0; i < repeat; ++i){
            res_simd = model.rbf_kernel_simd_(mat, mat);
        }
        start = GP::utils::print_time_spent(start);
        auto dif = res - res_simd;
        std::cout << "Naive vs SIMD: " << GP::utils::mean(dif * dif) << '\n';
    };
    GP::GPRegression model;
    for(auto& size_x : size_list){
        perform_bm(size_x, model);
    }
}



int main(int argc, const char* argv[]){
    using namespace GP::linalg;
    // train();
    // linalg_benchmark();
    // valid_matmul();
    // GP::matrix a, b;
    // std::cin >> a >> b;
    // std::cout << "a:\n" << a;
    // std::cout << "b:\n" << b;
    // GP::GPRegression model{0.003, 0.010};
    // model.fit(a, b);
    // auto&& [m, v] = model.predict(a);
    // std::cout << "m:\n" << m << "v:\n" << v;
    matmul_benchmark();
    inv_benchmark();
    kernel_benchmark();
    return 0;
}