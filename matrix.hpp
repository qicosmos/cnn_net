#pragma once
#include <cassert>
#include <stdexcept>

namespace cnn_net {
	class matrix {
	public:
		matrix() = default;
		matrix(matrix&& m) : data_(m.data_), rows_(m.rows_), cols_(m.cols_) {
		}

		~matrix() {
			delete[] data_;
		}

		matrix(const matrix& m) : rows_(m.rows_), cols_(m.cols_) {
			copy(m);
		}

		matrix& operator= (const matrix& m) {
			if (this == &m) {
				return *this;
			}

			if (rows_ != m.rows_ || cols_ != m.cols_) {
				delete[] data_;
				rows_ = m.rows_;
				cols_ = m.cols_;
				assert(rows_ != 0 && cols_ != 0);
				data_ = new double[rows_ * cols_]();
			}
			copy(m);

			return *this;
		}

		matrix& operator= (matrix&& m) {
			if (this == &m) {
				return *this;
			}

			delete[] data_;

			data_ = m.data_;
			rows_ = m.rows_;
			cols_ = m.cols_;
			m.reset();

			return *this;
		}

		matrix(unsigned rows, unsigned cols) : rows_(rows), cols_(cols) {
			if (rows == 0 || cols == 0)
				throw std::invalid_argument("Matrix constructor has 0 size");
			data_ = new double[rows * cols]();
		}

		matrix(unsigned rows) : rows_(rows), cols_(1) {
			if (rows == 0)
				throw std::invalid_argument("Matrix constructor has 0 size");
			data_ = new double[rows * cols_]();
		}

		double& operator() (unsigned row, unsigned col) {
			if (row >= rows_ || col >= cols_)
				throw std::invalid_argument("Matrix subscript out of bounds");
			return data_[cols_*row + col];
		}

		double  operator() (unsigned row, unsigned col) const {
			if (row >= rows_ || col >= cols_)
				throw std::invalid_argument("const Matrix subscript out of bounds");
			return data_[cols_*row + col];
		}

		double& operator() (unsigned row) {
			assert(cols_ == 1);
			if (row >= rows_)
				throw std::invalid_argument("Matrix subscript out of bounds");
			return data_[cols_*row];
		}

		double  operator() (unsigned row) const {
			assert(cols_ == 1);
			if (row >= rows_)
				throw std::invalid_argument("const Matrix subscript out of bounds");
			return data_[cols_*row];
		}

		size_t rows() const {
			return rows_;
		}

		size_t cols() const {
			return cols_;
		}

		size_t size() const {
			return rows_ * cols_;
		}

		void reset() {
			data_ = nullptr;
			rows_ = 0;
			cols_ = 0;
		}

	private:
		void copy(const matrix& m) {
			for (size_t i = 0; i < rows_*cols_; ++i) {
				data_[i] = m.data_[i];
			}
		}

	private:
		size_t rows_ = 0;
		size_t cols_ = 0;
		double* data_ = nullptr;;
	};

	using matrix_1d = matrix;
}