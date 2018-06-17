#pragma once
#include <vector>
#include "matrix.hpp"
#include "random.h"

namespace cnn_net {
	struct neuron {
		matrix_1d weights;//input 权值列表
		matrix_1d deltas; //delta列表
		double bias;      //偏置
		double output;	  //输出值

		void init(size_t input_size) {
			assert(input_size > 0);

			weights = matrix_1d(input_size);
			deltas = matrix_1d(input_size);

			for (size_t i = 0; i < input_size; i++) {
				weights(i) = uniform_rand<double>(-1, 1);
			}
		}
	};

	struct layer {
		std::vector<neuron> neurons;
		matrix_1d input;

		void init(size_t input_size, size_t neuron_count) {
			for (size_t i = 0; i < neuron_count; i++)
			{
				neuron n;
				n.init(input_size);
				neurons.push_back(n);
			}

			input = matrix_1d(input_size);
		}

		void calculate() {

		}
	};

	class bp_net {

	};
}