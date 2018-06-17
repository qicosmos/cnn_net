#include <iostream>
#include <valarray>

#include "bp_net.hpp"
using namespace cnn_net;

void print_mat(matrix& mat) {
	for (size_t r = 0; r < mat.rows(); r++)
	{
		for (size_t c = 0; c < mat.cols(); c++)
			std::cout << mat(r, c) << " ";
		std::cout << std::endl;
	}
}

int main(int argc, char** argv) {
	neuron nr;
	nr.init(3);
	matrix mat(3, 4);
	mat(0, 0) = 1;
	mat(0, 1) = 2;
	mat(1, 0) = 3;
	mat(1, 1) = 4;
	print_mat(mat);

	using matrix_1d = matrix;
	matrix_1d mat1(3);
	mat1(1) = 3;
	mat1(2) = 4;
	print_mat(mat1);
	int row = 3;
	int col = 4;
	std::valarray<double> matrix(row * col); // no more, no less, than a matrix
	matrix[std::slice(0, row, 0)] = 2;
	matrix[std::slice(1, row, 0)] = 3;
	matrix[std::slice(2, col, row)] = 3.14; // set third column to pi
	matrix[std::slice(3 * row, row, 1)] = 2.71; // set fourth row to e

	auto array = new double[10][10]();
	std::cout << typeid(array).name() << std::endl;
	array[0][0] = 2;
	array[1][0] = 3;
	array[0][1] = 4;
	array[1][1] = 5;
	for (int r = 0; r < 10; r++)
	{
		for (int c = 0; c < 10; c++)
			std::cout << array[r][c] << " ";
		std::cout << std::endl;
	}

	delete[] array;
	return 0;
}