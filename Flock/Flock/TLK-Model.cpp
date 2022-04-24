#include "TLK-Model.h"
#include <iostream>

//#define TLK_DEBUGMODE

//TODO:: comment implementation

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

		void LSTM(Compile, Model& m, size_t l)
		{
			std::vector<Gate>& gates = m.mlayers[l].gates;
			gates.push_back(Gate{}); // forget gate
			gates.push_back(Gate{}); // input gate
			gates.push_back(Gate{}); // cell gate
			gates.push_back(Gate{}); // output gate
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

		void LSTM(Agent, Model& m, ::TLK::Layer layer, size_t l, size_t count)
		{
			assert(layer.input.width > 0 && layer.output.width > 0);

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
	}

	Layer::Impl Dense{ 
		static_cast<void (*)(impl::Compute, Model&, Layer, size_t)>(impl::Dense),
		static_cast<void (*)(impl::Agent, Model&, Layer, size_t, size_t)>(impl::Dense),
		static_cast<void (*)(impl::Compile, Model&, size_t)>(impl::Dense)
	};
	Layer::Impl LSTM{ 
		static_cast<void (*)(impl::Compute, Model&, Layer, size_t)>(impl::LSTM),
		static_cast<void (*)(impl::Agent, Model&, Layer, size_t, size_t)>(impl::LSTM),
		static_cast<void (*)(impl::Compile, Model&, size_t)>(impl::LSTM)
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
			layers[i].impl.Compute({}, *this, layers[i], i);
	}
}