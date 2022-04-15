#include "TLK-Model.h"
#include <iostream>

namespace TLK
{
	namespace Compute
	{
		void Dense(Model& m, Layer l, size_t i)
		{
			MatLayer& curr = m.mlayers[i + 1];
			MatLayer& prev = m.mlayers[i];
			MatLayer& weights = m.weights[i];
			MatLayer& biases = m.biases[i];
			for (size_t n = 0; n < curr.size(); ++n)
			{
				// https://eigen.tuxfamily.org/dox/TopicWritingEfficientProductExpression.html => refer to 4th down the graph "m1.noalias() = m4 + m2 * m3;"
				curr[n] = biases[n];
				curr[n].noalias() += prev[n] * weights[n];

				curr[n] = curr[n].unaryExpr([](float val) { return tanh(val); });
			}
		}

		void LSTM(Model& m, Layer l, size_t i)
		{
			std::cout << "LSTM\n";
		}

		void Pooling(Model& m, Layer l, size_t i)
		{
			std::cout << "Pooling\n";
		}

		void Convolution(Model& m, Layer l, size_t i)
		{
			std::cout << "Convolution\n";
		}
	}

	namespace Append
	{
		void Dense(Model& m, Layer l, size_t i, size_t count)
		{
			assert(l.input.width > 0 && l.output.width > 0);

			for (; count > 0; --count)
			{
				if (i == 0) m.mlayers[0].push_back(Eigen::MatrixXf(1, l.input.width));
				m.mlayers[i + 1].push_back(Eigen::MatrixXf(1, l.output.width));
				m.weights[i].push_back(Eigen::MatrixXf::Random(l.input.width, l.output.width));
				m.biases[i].push_back(Eigen::MatrixXf::Random(1, l.output.width));
				//m.weights[i].push_back(Eigen::MatrixXf::Ones(l.input.width, l.output.width));
				//m.biases[i].push_back(Eigen::MatrixXf::Ones(1, l.output.width));
			}
		}

		void LSTM(Model& m, Layer l, size_t i, size_t count)
		{
		}

		void Pooling(Model& m, Layer l, size_t i, size_t count)
		{
		}

		void Convolution(Model& m, Layer l, size_t i, size_t count)
		{
		}
	}

	const Layer::Impl Dense{ Append::Dense, Compute::Dense };
	const Layer::Impl LSTM{ Append::LSTM, Compute::LSTM };
	const Layer::Impl Pooling{ Append::Pooling, Compute::Pooling };
	const Layer::Impl Convolution{ Append::Convolution, Compute::Convolution };

	void Model::Append(Layer layer)
	{
		assert(layers.size() == 0 || layers[layers.size() - 1].output == layer.input);

		layers.push_back(layer);
	}

	void Model::Compile()
	{
		mlayers = new MatLayer[layers.size() + 1];
		weights = new MatLayer[layers.size()];
		biases = new MatLayer[layers.size()];
	}

	void Model::Agent(size_t count)
	{
		assert(mlayers != nullptr && weights != nullptr && biases != nullptr);

		for (size_t i = 0; i < layers.size(); i++)
		{
			layers[i].methods.Append(*this, layers[i], i, count);
		}
	}

	void Model::Compute()
	{
		assert(mlayers != nullptr && weights != nullptr && biases != nullptr);

		for (int i = 0; i < layers.size(); i++)
		{
			layers[i].methods.Compute(*this, layers[i], i);
		}
	}
}