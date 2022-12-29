#include <matrix/matrix.hpp>
#include <linalg/linalg.hpp>
#include <utils.hpp>
#include <gp/model.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

auto print_time_spent(std::chrono::high_resolution_clock::time_point start_time){
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = end_time - start_time;
    std::cout << "Time spent: " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
        << " (ms)"
        << std::endl;
    return end_time;
}

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


void preprocess(GP::matrix& m, const GP::matrix& mu, GP::matrix& stdv){
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
        start = print_time_spent(start);
        auto&& [mu, var] = model.predict(X_test);
        std::cout << "[Predict] ";
        start = print_time_spent(start);

        // mse 
        auto diff = (mu - Y_test);
        std::cout << "MSE: " << (transpose(diff) ^ diff) * (1./diff.size())<<"\n";

        auto compare = GP::matrix{Y_test.size(), 2};
        for(size_t idx = 0; idx < Y_test.size(); ++idx){
            compare(idx, 0) = mu(idx, 0);
            compare(idx, 1) = Y_test(idx, 0);
        }
    }
}

void linalg_benchmark(){
    using namespace GP::linalg;
    GP::matrix A{randn(100, 100)}, B{randn(300, 300)}, C{randn(1000, 1000)};
    int repeat = 5;

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "[matmul] Repeat " << repeat <<" times \n";
    std::cout << "A 100x100 - ";
    for(int i = 0; i < repeat; ++i)
        A ^= A;
    start = print_time_spent(start);

    std::cout << "B 300x300 - ";
    for(int i = 0; i < repeat; ++i)
        B ^= B;
    start = print_time_spent(start);
    
    std::cout << "C 1000x1000 - ";
    for(int i = 0; i < repeat; ++i)
        C ^= C;
    start = print_time_spent(start);

    std::cout << "[inv] Repeat " << repeat <<" times \n";
        std::cout << "A 100x100 - ";
    for(int i = 0; i < repeat; ++i)
        A = ~A;
    start = print_time_spent(start);

    std::cout << "B 300x300 - ";
    for(int i = 0; i < repeat; ++i)
        B = ~B;
    start = print_time_spent(start);
    
    std::cout << "C 1000x1000 - ";
    for(int i = 0; i < repeat; ++i)
        C = ~C;
    start = print_time_spent(start);
}

int main(int argc, const char* argv[]){
    using namespace GP::linalg;
    train();
    // linalg_benchmark();
    return 0;
}