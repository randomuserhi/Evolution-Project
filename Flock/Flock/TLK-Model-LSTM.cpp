#include "TLK-Model.h"
#include <iostream>

namespace TLK
{
	namespace impl
	{
		void LSTM(Compile, Model& m, ::TLK::Layer layer, size_t l)
		{
			assert(layer.input.width > 0 && layer.output.width > 0);

			std::vector<Gate>& gates = m.mlayers[l].gates;
			gates.push_back(Gate{}); // forget gate
			gates.push_back(Gate{}); // input gate
			gates.push_back(Gate{}); // cell gate
			gates.push_back(Gate{}); // output gate
		}

		void LSTM(Agent, Model& m, ::TLK::Layer layer, size_t l, size_t count)
		{
			Layer& mlayer = m.mlayers[l];

			for (size_t i = 0, index = m.count; i < count; ++i, ++index)
			{
				if (l != 0)
					mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.mlayers[l - 1].outputs[index].data(), 1, layer.input.width));
				else
				{
					m.inputs.push_back(Eigen::MatrixXf::Zero(1, layer.input.width));
					mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.inputs[index].data(), 1, layer.input.width));
				}

				mlayer.buffer.push_back(Eigen::MatrixXf::Zero(1, layer.input.width + layer.output.width));
				mlayer.state.push_back(Eigen::MatrixXf::Zero(1, layer.output.width));

				for (size_t g = 0; g < 4; ++g)
				{
					mlayer.gates[g].nodes.push_back(Eigen::MatrixXf::Zero(1, layer.output.width));

#ifdef TLK_DEBUGMODE
					mlayer.gates[g].weights.push_back(Eigen::MatrixXf::Ones(layer.input.width + layer.output.width, layer.output.width));
					mlayer.gates[g].biases.push_back(Eigen::MatrixXf::Ones(1, layer.output.width));
#else
					mlayer.gates[g].weights.push_back(Eigen::MatrixXf::Random(layer.input.width + layer.output.width, layer.output.width));
					mlayer.gates[g].biases.push_back(Eigen::MatrixXf::Random(1, layer.output.width));
#endif
				}

				mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.buffer[index].data() + layer.input.width, 1, layer.output.width));
				if (l == m.layers.size() - 1)
					m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.buffer[index].data() + layer.input.width, 1, layer.output.width));
			}
		}

		void LSTM(Recompile, Model& m, ::TLK::Layer layer, size_t l)
		{
			Layer& mlayer = m.mlayers[l];

			for (size_t i = 0; i < m.count; ++i)
			{
				if (l != 0)
					mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.mlayers[l - 1].outputs[i].data(), 1, layer.input.width));
				else
				{
					m.inputs.push_back(Eigen::MatrixXf::Zero(1, layer.input.width));
					mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.inputs[i].data(), 1, layer.input.width));
				}

				mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.buffer[i].data() + layer.input.width, 1, layer.output.width));
				if (l == m.layers.size() - 1)
					m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.buffer[i].data() + layer.input.width, 1, layer.output.width));
			}
		}

		void LSTM(Duplicate, Model& m, ::TLK::Layer layer, size_t l, size_t index)
		{
			Layer& mlayer = m.mlayers[l];

			if (l != 0)
				mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.mlayers[l - 1].outputs[m.count].data(), 1, layer.input.width));
			else
			{
				m.inputs.push_back(Eigen::MatrixXf::Zero(1, layer.input.width));
				mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.inputs[m.count].data(), 1, layer.input.width));
			}

			mlayer.buffer.push_back(Eigen::MatrixXf::Zero(1, layer.input.width + layer.output.width));
			mlayer.state.push_back(Eigen::MatrixXf::Zero(1, layer.output.width));

			for (size_t g = 0; g < 4; ++g)
			{
				mlayer.gates[g].nodes.push_back(Eigen::MatrixXf::Zero(1, layer.output.width));

				mlayer.gates[g].weights.push_back(Eigen::MatrixXf(layer.input.width + layer.output.width, layer.output.width));
				mlayer.gates[g].biases.push_back(Eigen::MatrixXf(1, layer.output.width));

				memcpy(mlayer.gates[g].weights[m.count].data(), mlayer.gates[g].weights[index].data(), sizeof(float) * mlayer.gates[g].weights[m.count].size());
				memcpy(mlayer.gates[g].biases[m.count].data(), mlayer.gates[g].biases[index].data(), sizeof(float) * mlayer.gates[g].biases[m.count].size());
			}

			mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.buffer[m.count].data() + layer.input.width, 1, layer.output.width));
			if (l == m.layers.size() - 1)
				m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.buffer[m.count].data() + layer.input.width, 1, layer.output.width));
		}

		void LSTM(Compute, Model& m, ::TLK::Layer layer, size_t l)
		{
			Layer& mlayer = m.mlayers[l];
			std::vector<Eigen::Map<Eigen::MatrixXf>>& inputs = mlayer.inputs;
			std::vector<Eigen::Map<Eigen::MatrixXf>>& hidden = mlayer.outputs;
			std::vector<Eigen::MatrixXf>& buffer = mlayer.buffer;
			std::vector<Eigen::MatrixXf>& cell = mlayer.state;
			Gate& forgetGate = mlayer.gates[0];
			Gate& inputGate = mlayer.gates[1];
			Gate& cellGate = mlayer.gates[2];
			Gate& outputGate = mlayer.gates[3];

			for (size_t i = 0; i < m.count; ++i)
			{
				std::memcpy(buffer[i].data(), inputs[i].data(), sizeof(float) * layer.input.width);

				// https://eigen.tuxfamily.org/dox/TopicWritingEfficientProductExpression.html => refer to 4th down the graph "m1.noalias() = m4 + m2 * m3;"

				forgetGate.nodes[i] = forgetGate.biases[i];
				inputGate.nodes[i] = inputGate.biases[i];
				cellGate.nodes[i] = cellGate.biases[i];
				outputGate.nodes[i] = outputGate.biases[i];

				forgetGate.nodes[i].noalias() += buffer[i] * forgetGate.weights[i];
				inputGate.nodes[i].noalias() += buffer[i] * inputGate.weights[i];
				cellGate.nodes[i].noalias() += buffer[i] * cellGate.weights[i];
				outputGate.nodes[i].noalias() += buffer[i] * outputGate.weights[i];

				forgetGate.nodes[i] = forgetGate.nodes[i].unaryExpr([](float val) { float Exp = exp(val); return Exp / (Exp + 1); });
				inputGate.nodes[i] = inputGate.nodes[i].unaryExpr([](float val) { float Exp = exp(val); return Exp / (Exp + 1); });
				outputGate.nodes[i] = outputGate.nodes[i].unaryExpr([](float val) { float Exp = exp(val); return Exp / (Exp + 1); });
				cellGate.nodes[i] = cellGate.nodes[i].unaryExpr([](float val) { return tanh(val); });

				cell[i].noalias() = cell[i].cwiseProduct(forgetGate.nodes[i]);
				cell[i].noalias() += inputGate.nodes[i].cwiseProduct(cellGate.nodes[i]);

				hidden[i] = cell[i].unaryExpr([](float val) { return tanh(val); });
				hidden[i].noalias() = hidden[i].cwiseProduct(outputGate.nodes[i]);
			}
		}

		void LSTM(Remove, Model& m, ::TLK::Layer layer, size_t l, size_t i)
		{
			Layer& mlayer = m.mlayers[l];

			mlayer.buffer.erase(mlayer.buffer.begin() + i);
			mlayer.state.erase(mlayer.state.begin() + i);

			for (size_t g = 0; g < 4; ++g)
			{
				mlayer.gates[g].nodes.erase(mlayer.gates[g].nodes.begin() + i);
				mlayer.gates[g].weights.erase(mlayer.gates[g].weights.begin() + i);
				mlayer.gates[g].biases.erase(mlayer.gates[g].biases.begin() + i);
			}
		}

		void LSTM(Mutate, Model& m, ::TLK::Layer layer, size_t l, size_t i)
		{
			Layer& mlayer = m.mlayers[l];
			for (size_t j = 0; j < 4; ++j)
			{
				Gate& g = mlayer.gates[j];
				g.weights[i] = g.weights[i].unaryExpr([](float val)
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
				g.biases[i] = g.biases[i].unaryExpr([](float val)
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
			}
		}
	}

	Layer::Impl LSTM{
		static_cast<void (*)(impl::Duplicate, Model&, Layer, size_t, size_t)>(impl::LSTM),
		static_cast<void (*)(impl::Compute, Model&, Layer, size_t)>(impl::LSTM),
		static_cast<void (*)(impl::Mutate, Model&, Layer, size_t, size_t)>(impl::LSTM),
		static_cast<void (*)(impl::Agent, Model&, Layer, size_t, size_t)>(impl::LSTM),
		static_cast<void (*)(impl::Remove, Model&, Layer, size_t, size_t)>(impl::LSTM),
		static_cast<void (*)(impl::Compile, Model&, Layer, size_t)>(impl::LSTM),
		static_cast<void (*)(impl::Recompile, Model&, Layer, size_t)>(impl::LSTM)
	};
}