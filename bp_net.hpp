#pragma once
#include <vector>
#include "matrix.hpp"
#include "random.h"

namespace cnn_net {
	struct neuron {
		neuron() = default;
		neuron(size_t input_size) {
			init(input_size);
		}
		neuron(const neuron& n) = delete;
		/*{
			weights = n.weights;
			deltas = n.deltas;
			bias = n.bias;
			output = n.output;
		}*/
		neuron(neuron&& n) : weights(std::move(n.weights)), deltas(std::move(n.deltas)),
			bias(n.bias), output(n.output)
		{
			n.reset();
		}

		neuron& operator= (const neuron& m) = delete;
		neuron& operator= (neuron&& m) = delete;

		matrix_1d weights;//input 权值列表
		matrix_1d deltas; //delta列表
		double bias = 0;      //偏置
		double output = 0;	  //输出值

		void init(size_t input_size) {
			assert(input_size > 0);

			weights = { input_size };
			deltas = matrix_1d(input_size);

			for (size_t i = 0; i < input_size; i++) {
				weights(i) = uniform_rand<double>(-1, 1);
			}
		}

		void reset() {
			weights.reset();
			deltas.reset();
			bias = 0;
			output = 0;
		}
	};

	struct layer {
		std::vector<neuron> neurons; //the next layer neurons
		matrix_1d input; //input values

		layer() = default;
		layer(size_t input_size, size_t neuron_count) {
			init(input_size, neuron_count);
		}

		void init(size_t input_size, size_t neuron_count) {
			for (size_t i = 0; i < neuron_count; i++)
			{
				//neuron n(input_size);
				neurons.emplace_back(neuron{ input_size });
			}

			input = matrix_1d(input_size);

			//just for test
			for (size_t i = 0; i < input_size; i++) {
				input(i) = uniform_rand<double>(-1, 1);
			}
		}

		void calculate() {
			double sum = 0;
			for (size_t i = 0; i < neurons.size(); i++)
			{
				for (size_t j = 0; j < input.size(); j++)
				{
					sum += input(j)*neurons[i].weights(j);
				}

				sum += neurons[i].bias;
				neurons[i].output = 1.f / (1.f + exp(-sum)); //sigmoid
			}
		}
	};

	class bp_net {
	public:
		void init(size_t input_count, size_t input_neurons, size_t output_count, std::vector<size_t> hidden_neurons) {
			assert(input_count && input_neurons && output_count && !hidden_neurons.empty());

			//init input layer
			input_layer_.init(input_count, input_neurons);

			//init hidden layers
			for (size_t i = 0; i < hidden_neurons.size(); i++){
				if(i==0){
					hidden_layers_[i] = { input_count, hidden_neurons[0] };
				}
				else {
					hidden_layers_[i] = { hidden_neurons[i-1], hidden_neurons[i] };
				}
			}

			//init output layer
			output_layer_.init(hidden_neurons[hidden_neurons.size() - 1], output_count);
		}

		void update(int layer_index) {
			if (layer_index == -1) {
				for (size_t i = 0; i < input_layer_.neurons.size(); i++)
				{
					hidden_layers_[0].input(i) = input_layer_.neurons[i].output;
				}
			}
			else {
				for (size_t i = 0; i < hidden_layers_[layer_index].neurons.size(); i++)
				{
					if ((size_t)layer_index < hidden_layers_.size() - 1) {
						hidden_layers_[layer_index + 1].input(i) = hidden_layers_[layer_index].neurons[i].output;
					}
					else {
						output_layer_.input(i) = hidden_layers_[layer_index].neurons[i].output;
					}
				}
			}
		}

		void forward(const double *input) {
			input_layer_.calculate();
			update(-1);

			for (size_t i = 0; i < hidden_layers_.size(); i++)
			{
				hidden_layers_[i].calculate();
				update(i);
			}
		}

		double train(const double *desired_output, const double *input, double alpha, double momentum) {
			//todo
			return 0;
		}

		layer &get_output_layer()
		{
			return output_layer_;
		}

	private:
		layer input_layer_;
		layer output_layer_;
		std::vector<layer> hidden_layers_;
	};
}