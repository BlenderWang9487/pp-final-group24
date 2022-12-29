#pragma once
#include <algorithm>
#include <memory>
#include <cstring>
#include <utility>
#include <iostream>
#include <iomanip>


namespace GP{

class matrix{
private:
    std::shared_ptr<double[]> buffer_;
    size_t row_;
    size_t col_;
    size_t pad_col_;
    static constexpr const size_t simd_len = 4;
    static inline size_t aligned_size(size_t s){
        return (s + (simd_len-1))/simd_len*simd_len;
    }
public:
    static size_t copy_count;
    struct DeleteAligned
    {
        void operator()(double* data) const
        {
            free(data);
        }
    };
    class DimensionalityException : public std::exception{
    public:
        const char* what() const throw() { 
            return "GP::matrix dimensions don't match.";
        } 
    };

    static void matrix_get_copied(size_t space){
        // this function is for debug
        std::cout << "Matrix getting copied. "<< space<<" items are copied.\n";
        copy_count+=space;
    }

    // constructor
    matrix():
        row_{1}, col_{1}, pad_col_{4}, buffer_((double*)aligned_alloc(simd_len * sizeof(double), aligned_size(1) * sizeof(double)), DeleteAligned())
        {}
    matrix(size_t r, size_t c):
        row_{r}, col_{c}, pad_col_{aligned_size(c)}, buffer_((double*)aligned_alloc(simd_len * sizeof(double), r*aligned_size(c) * sizeof(double)), DeleteAligned())
        {
            if(r * c == 0){
                throw DimensionalityException();
            }
            std::memset(buffer_.get(), 0, this->pad_size() * sizeof(double));
        }
    matrix(size_t n):
        row_{n}, col_{n}, pad_col_{aligned_size(n)}, buffer_((double*)aligned_alloc(simd_len * sizeof(double), n*aligned_size(n) * sizeof(double)), DeleteAligned())
        {
            if(n == 0){
                throw DimensionalityException();
            }
            std::memset(buffer_.get(), 0, this->pad_size() * sizeof(double));
        }
    matrix(const matrix& m):
        row_{m.row_}, col_{m.col_}, pad_col_{m.pad_col_}, buffer_((double*)aligned_alloc(simd_len * sizeof(double), m.row_*m.pad_col_ * sizeof(double)), DeleteAligned())
    {
        size_t buffer_size = row_*pad_col_;
        std::memcpy(
            buffer_.get(),
            m.buffer_.get(),
            buffer_size*sizeof(double));
        // matrix_get_copied(buffer_size);
    }
    matrix(matrix&& m):
        row_{m.row_}, col_{m.col_}, pad_col_{m.pad_col_}, buffer_(m.buffer_)
    {}

    // assign
    matrix& operator=(const matrix& m){
        row_ = m.row_; col_ = m.col_; pad_col_ = m.pad_col_;
        size_t buffer_size = row_*pad_col_;
        buffer_.reset((double*)aligned_alloc(simd_len * sizeof(double), buffer_size * sizeof(double)), DeleteAligned());
        std::memcpy(
            buffer_.get(),
            m.buffer_.get(),
            buffer_size*sizeof(double));
        // matrix_get_copied(buffer_size);
        return *this;
    }
    matrix& operator=(matrix&& m){
        row_ = m.row_; col_ = m.col_; pad_col_ = m.pad_col_;
        buffer_ = m.buffer_;
        return *this;
    }

    inline auto size() const -> size_t {
        return row_*col_;
    }
    inline auto pad_size() const -> size_t {
        return row_*pad_col_;
    }
    inline auto pad_column() const -> size_t {
        return pad_col_;
    }
    inline auto shape() const -> std::pair<size_t, size_t> {
        return std::pair<size_t, size_t>{row_, col_};
    }
    auto shape(size_t dim) const -> size_t {
        switch (dim)
        {
        case 0:
            return row_;
            break;
        case 1:
            return col_;
            break;
        
        default:
            throw DimensionalityException();
        }
    }
    inline auto ptr() const {
        return buffer_.get();
    }
    inline double& operator()(size_t r, size_t c){
        return buffer_[r*pad_col_ + c];
    }
    inline double operator()(size_t r, size_t c) const {
        return buffer_[r*pad_col_ + c];
    }

    // operator
    matrix operator+(double val){
        matrix res{row_, col_};
        for(size_t r = 0; r < row_; ++ r)
            for(size_t c = 0; c < col_; ++ c)
                res.buffer_[r*pad_col_ + c] = buffer_[r*pad_col_ + c] + val;
        return res;
    }
    matrix operator+(const matrix& rhs){
        auto&& [rrow, rcol] = rhs.shape();
        if(row_ != rrow or col_ != rcol){
            throw matrix::DimensionalityException();
        }
        matrix res{row_, col_};
        for(size_t r = 0; r < row_; ++ r)
            for(size_t c = 0; c < col_; ++ c)
                res.buffer_[r*pad_col_ + c] = buffer_[r*pad_col_ + c] + rhs.buffer_[r*pad_col_ + c];
        return res;
    }
    inline matrix operator-(){
        return (*this)*double{-1};
    }
    inline matrix operator-(double val){
        return (*this) + (-val);
    }
    matrix operator-(const matrix& rhs){
        auto&& [rrow, rcol] = rhs.shape();
        if(row_ != rrow or col_ != rcol){
            throw matrix::DimensionalityException();
        }
        matrix res{row_, col_};
        for(size_t r = 0; r < row_; ++ r)
            for(size_t c = 0; c < col_; ++ c)
                res.buffer_[r*pad_col_ + c] = buffer_[r*pad_col_ + c] - rhs.buffer_[r*pad_col_ + c];
        return res;
    }
    matrix operator*(double val){
        matrix res{row_, col_};
        for(size_t r = 0; r < row_; ++ r)
            for(size_t c = 0; c < col_; ++ c)
                res.buffer_[r*pad_col_ + c] = buffer_[r*pad_col_ + c] * val;
        return res;
    }
    matrix operator*(const matrix& rhs){
        auto&& [rrow, rcol] = rhs.shape();
        if(row_ != rrow or col_ != rcol){
            throw matrix::DimensionalityException();
        }
        matrix res{row_, col_};
        for(size_t r = 0; r < row_; ++ r)
            for(size_t c = 0; c < col_; ++ c)
                res.buffer_[r*pad_col_ + c] = buffer_[r*pad_col_ + c] * rhs.buffer_[r*pad_col_ + c];
        return res;
    }

    // any $= operator
    matrix& operator=(double val){
        for(size_t r = 0; r < row_; ++ r)
            for(size_t c = 0; c < col_; ++ c)
                buffer_[r*pad_col_ + c] = val;
        return *this;
    }
    matrix& operator+=(double val){
        for(size_t r = 0; r < row_; ++ r)
            for(size_t c = 0; c < col_; ++ c)
                buffer_[r*pad_col_ + c] += val;
        return *this;
    }
    inline matrix& operator-=(double val){
        return (*this) += -val;
    }
    matrix& operator*=(double val){
        for(size_t r = 0; r < row_; ++ r)
            for(size_t c = 0; c < col_; ++ c)
                buffer_[r*pad_col_ + c] *= val;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream &os, const matrix& m){
        for(size_t r = 0; r < m.row_;++r){
            for(size_t c = 0; c < m.col_;++c)
                os << std::setprecision(3) << std::fixed << m(r, c) << '\t';
            os << '\n';
        }
        return os;
    }
    friend std::istream& operator>>(std::istream &is, matrix& m){
        size_t row, col;
        is >> row >> col;
        m = matrix{row, col};
        for(size_t r = 0; r < row; ++r)
            for(size_t c = 0; c < col; ++c)
                is >> m(r, c);
        return is;
    }
};
size_t matrix::copy_count = 0;

}