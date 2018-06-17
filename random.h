#pragma once
#include <limits>
#include <random>
#include <type_traits>

namespace cnn_net {
	class random_generator {
	public:
		static random_generator &get_instance() {
			static random_generator instance;
			return instance;
		}

		std::mt19937 &operator()() { return gen_; }

		void set_seed(unsigned int seed) { gen_.seed(seed); }

	private:
		// avoid gen_(0) for MSVC known issue
		// https://connect.microsoft.com/VisualStudio/feedback/details/776456
		random_generator() : gen_(1) {}
		std::mt19937 gen_;
	};

	template <typename T>
	inline typename std::enable_if<std::is_integral<T>::value, T>::type
		uniform_rand(T min, T max) {
		std::uniform_int_distribution<T> dst(min, max);
		return dst(random_generator::get_instance()());
	}

	template <typename T>
	inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
		uniform_rand(T min, T max) {
		std::uniform_real_distribution<T> dst(min, max);
		return dst(random_generator::get_instance()());
	}

	template <typename T>
	inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
		gaussian_rand(T mean, T sigma) {
		std::normal_distribution<T> dst(mean, sigma);
		return dst(random_generator::get_instance()());
	}

	inline void set_random_seed(unsigned int seed) {
		random_generator::get_instance().set_seed(seed);
	}
}