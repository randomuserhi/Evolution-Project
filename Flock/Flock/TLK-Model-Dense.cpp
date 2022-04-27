#include "TLK-Model.h"
#include <iostream>

namespace TLK
{
	namespace impl
	{
		void Dense(Compile, Model& m, ::TLK::Layer layer, size_t l)
		{
			assert(layer.input.width > 0 && layer.output.width > 0);

			std::vector<Gate>& gates = m.mlayers[l].gates;
			gates.push_back(Gate{});
		}

		void Dense(Agent, Model& m, ::TLK::Layer layer, size_t l, size_t count)
		{
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
#ifdef TLK_DEBUGMODE
				g.weights.push_back(Eigen::MatrixXf::Ones(layer.input.width, layer.output.width));
				g.biases.push_back(Eigen::MatrixXf::Ones(1, layer.output.width));
#else
				g.weights.push_back(Eigen::MatrixXf::Random(layer.input.width, layer.output.width));
				g.biases.push_back(Eigen::MatrixXf::Random(1, layer.output.width));
#endif

				mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(g.nodes[index].data(), 1, layer.output.width));
				if (l == m.layers.size() - 1)
					m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(g.nodes[index].data(), 1, layer.output.width));
			}
		}

		void Dense(Recompile, Model& m, ::TLK::Layer layer, size_t l)
		{
			Layer& mlayer = m.mlayers[l];
			Gate& g = mlayer.gates[0];

			for (size_t i = 0; i < m.count; ++i)
			{
				if (l != 0)
					mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.mlayers[l - 1].outputs[i].data(), 1, layer.input.width));
				else
				{
					m.inputs.push_back(Eigen::MatrixXf::Zero(1, layer.input.width));
					mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.inputs[i].data(), 1, layer.input.width));
				}

				mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(g.nodes[i].data(), 1, layer.output.width));
				if (l == m.layers.size() - 1)
					m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(g.nodes[i].data(), 1, layer.output.width));
			}
		}

		void Dense(Duplicate, Model& m, ::TLK::Layer layer, size_t l, size_t index)
		{
			Layer& mlayer = m.mlayers[l];
			Gate& g = mlayer.gates[0];

			if (l != 0)
				mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.mlayers[l - 1].outputs[m.count].data(), 1, layer.input.width));
			else
			{
				m.inputs.push_back(Eigen::MatrixXf::Zero(1, layer.input.width));
				mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.inputs[m.count].data(), 1, layer.input.width));
			}

			g.nodes.push_back(Eigen::MatrixXf::Zero(1, layer.output.width));

			g.weights.push_back(Eigen::MatrixXf(layer.input.width, layer.output.width));
			g.biases.push_back(Eigen::MatrixXf(1, layer.output.width));

			memcpy(g.weights[m.count].data(), g.weights[index].data(), sizeof(float) * g.weights[m.count].size());
			memcpy(g.biases[m.count].data(), g.biases[index].data(), sizeof(float) * g.biases[m.count].size());

			mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(g.nodes[m.count].data(), 1, layer.output.width));
			if (l == m.layers.size() - 1)
				m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(g.nodes[m.count].data(), 1, layer.output.width));
		}

		void Dense(Compute, Model& m, ::TLK::Layer layer, size_t l)
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

		void Dense(Remove, Model& m, ::TLK::Layer layer, size_t l, size_t index)
		{
			Layer& mlayer = m.mlayers[l];
			Gate& g = mlayer.gates[0];

			g.nodes.erase(g.nodes.begin() + index);
			g.biases.erase(g.biases.begin() + index);
			g.weights.erase(g.weights.begin() + index);
		}

		void Dense(Mutate, Model& m, ::TLK::Layer layer, size_t l, size_t i)
		{
			Layer& mlayer = m.mlayers[l];
			Gate& g = mlayer.gates[0];
			std::vector<Eigen::MatrixXf>& weights = g.weights;
			std::vector<Eigen::MatrixXf>& biases = g.biases;

			weights[i] = weights[i].unaryExpr([](float val)
				{
					float chance = std::rand() / float(RAND_MAX);
					if (chance < 0.003)
						return val * -1;
					else if (chance < 0.006)
						return val + std::rand() / float(RAND_MAX);
					else if (chance < 0.009)
					{
						return val - std::rand() / float(RAND_MAX);
					}
					return val + (std::rand() / float(RAND_MAX) * 2.0f - 1.0f) * 0.01f;
				});
			biases[i] = biases[i].unaryExpr([](float val)
				{
					float chance = std::rand() / float(RAND_MAX);
					if (chance < 0.003f)
						return val * -1;
					else if (chance < 0.006f)
						return val + std::rand() / float(RAND_MAX);
					else if (chance < 0.009f)
					{
						return val - std::rand() / float(RAND_MAX);
					}
					return val + (std::rand() / float(RAND_MAX) * 2.0f - 1.0f) * 0.01f;
				});
		}
	}

	Layer::Impl Dense{
		static_cast<void (*)(impl::Duplicate, Model&, Layer, size_t, size_t)>(impl::Dense),
		static_cast<void (*)(impl::Compute, Model&, Layer, size_t)>(impl::Dense),
		static_cast<void (*)(impl::Mutate, Model&, Layer, size_t, size_t)>(impl::Dense),
		static_cast<void (*)(impl::Agent, Model&, Layer, size_t, size_t)>(impl::Dense),
		static_cast<void (*)(impl::Remove, Model&, Layer, size_t, size_t)>(impl::Dense),
		static_cast<void (*)(impl::Compile, Model&, Layer, size_t)>(impl::Dense),
		static_cast<void (*)(impl::Recompile, Model&, Layer, size_t)>(impl::Dense)
	};
}