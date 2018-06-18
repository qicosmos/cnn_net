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
	
	layer l;
	l.init(3, 2);
	l.calculate();
	
	return 0;
}