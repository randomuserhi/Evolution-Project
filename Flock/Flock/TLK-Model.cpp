#include "TLK-Model.h"
#include <iostream>

namespace TLK
{
	namespace impl
	{
		Compile compile{};
		Compile compute{};
		Agent append{};

		void Dense(Compile, Model& m, size_t l)
		{
			std::vector<Gate>& gates = m.mlayers[l].gates;
			gates.push_back(Gate{});
		}

		void Dense(Compute, Model& m, size_t l)
		{
			Layer& mlayer = m.mlayers[l];
			std::vector<Eigen::Map<Eigen::MatrixXf>>& inputs = mlayer.inputs;
			Gate& g = mlayer.gates[0];
			std::vector<Eigen::MatrixXf>& nodes = g.nodes;
			std::vector<Eigen::MatrixXf>& weights = g.weights;
			std::vector<Eigen::MatrixXf>& biases = g.biases;

			for (size_t i = 0; i < m.count; ++i)
			{
				// https://eigen.tuxfamily.org/dox/TopicWritingEfficientProductExpression.html => refer to 4th down the graph "m1.noalias() = m4 + m2 * m3;"
				nodes[i] = biases[i];
				nodes[i].noalias() += inputs[i] * weights[i];

				nodes[i] = nodes[i].unaryExpr([](float val) { return tanh(val); });
			}
		}

		void Dense(Agent, Model& m, ::TLK::Layer layer, size_t l, size_t count)
		{
			assert(layer.input.width > 0 && layer.output.width > 0);

			Layer& mlayer = m.mlayers[l];
			Gate& g = mlayer.gates[0];

			for (size_t i = 0, index = m.count; i < count; ++i, ++index)
			{
				if (l != 0)
					mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.mlayers[l - 1].outputs[index].data(), 1, layer.input.width));
				else
				{
					m.inputs.push_back(Eigen::MatrixXf::Zero(1, layer.input.width));
					mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.inputs[index].data(), 1, layer.input.width));
				}

				g.nodes.push_back(Eigen::MatrixXf::Zero(1, layer.output.width));
				g.weights.push_back(Eigen::MatrixXf::Random(layer.input.width, layer.output.width));
				g.biases.push_back(Eigen::MatrixXf::Random(1, layer.output.width));

				mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(g.nodes[index].data(), 1, layer.output.width));
				if (l == m.layers.size() - 1)
					m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(g.nodes[index].data(), 1, layer.output.width));
			}
		}
	}

	Layer::Impl Dense{ 
		static_cast<void (*)(impl::Compute, Model&, size_t)>(impl::Dense),
		static_cast<void (*)(impl::Agent, Model&, Layer, size_t, size_t)>(impl::Dense),
		static_cast<void (*)(impl::Compile, Model&, size_t)>(impl::Dense)
	};
	Layer::Impl LSTM{ 
		nullptr, 
		nullptr, 
		nullptr 
	};
	Layer::Impl Convolution{ 
		nullptr, 
		nullptr, 
		nullptr 
	};

	void Model::Compile()
	{
		mlayers.reserve(layers.size());

		for (size_t i = 0; i < layers.size(); ++i)
		{
			mlayers.push_back(impl::Layer{});
			layers[i].impl.Compile({}, *this, i);
		}

		compiled = true;
	}

	void Model::Append(Layer layer)
	{
		assert(!compiled);
		assert(layers.size() == 0 || (layers.size() != 0 && layers.back().output == layer.input));

		layers.push_back(layer);
	}

	void Model::Agent(size_t count)
	{
		assert(compiled);

		for (size_t i = 0; i < layers.size(); ++i)
			layers[i].impl.Agent({}, *this, layers[i], i, count);

		this->count += count;
	}

	void Model::Compute()
	{
		assert(compiled);

		for (size_t i = 0; i < layers.size(); ++i)
			layers[i].impl.Compute({}, *this, i);
	}
}