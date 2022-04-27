#include "TLK-Model.h"
#include <iostream>

//#define TLK_DEBUGMODE

//TODO:: comment implementation
//
//Convolution layers are too slow performance wise to use in my application! D:

//NOTE:: std::vector erase() seems to move Eigen::Map around causing problems so recompile function has to be used to remake all the maps.

namespace TLK
{
	namespace impl
	{
		Compile compile{};
		Compile compute{};
		Agent append{};
		Remove remove{};
		Recompile recompile{};
		Mutate mutate{};
		Duplicate duplicate{};

		void Dense(Compile, Model& m, ::TLK::Layer layer, size_t l)
		{
			assert(layer.input.width > 0 && layer.output.width > 0);

			std::vector<Gate>& gates = m.mlayers[l].gates;
			gates.push_back(Gate{});
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

		void Dense(Remove, Model& m, ::TLK::Layer layer, size_t l, size_t index)
		{
			Layer& mlayer = m.mlayers[l];
			Gate& g = mlayer.gates[0];

			g.nodes.erase(g.nodes.begin() + index);
			g.biases.erase(g.biases.begin() + index);
			g.weights.erase(g.weights.begin() + index);
		}

		void Dense(Recompile, Model& m, ::TLK::Layer layer, size_t l)
		{
			Layer& mlayer = m.mlayers[l];
			Gate& g = mlayer.gates[0];

			mlayer.inputs.clear();
			mlayer.outputs.clear();

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

		void LSTM(Compile, Model& m, ::TLK::Layer layer, size_t l)
		{
			assert(layer.input.width > 0 && layer.output.width > 0);

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

		void LSTM(Recompile, Model& m, ::TLK::Layer layer, size_t l)
		{
			Layer& mlayer = m.mlayers[l];

			mlayer.inputs.clear();
			mlayer.outputs.clear();

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

		void Convolution(Compile, Model& m, ::TLK::Layer layer, size_t l)
		{
			assert(layer.input.width > 0 && layer.input.height > 0 && layer.input.depth > 0);
			assert(layer.output.width > 0 && layer.output.height > 0 && layer.output.depth > 0);
			assert(layer.filter.width > 0 && layer.filter.height > 0 && layer.filter.depth > 0);
			assert(layer.filter.depth == layer.input.depth);
			assert(layer.strideX > 0 && layer.strideY > 0 && layer.zeroPadding >= 0);

			std::vector<Gate>& gates = m.mlayers[l].gates;
			gates.push_back({});
		}

		void Convolution(Compute, Model& m, ::TLK::Layer layer, size_t l)
		{
			Layer& mlayer = m.mlayers[l];
			std::vector<Eigen::Map<Eigen::MatrixXf>>& inputs = mlayer.inputs;
			std::vector<Eigen::Map<Eigen::MatrixXf>>& hidden = mlayer.outputs;
			std::vector<Eigen::MatrixXf>& buffer = mlayer.buffer;
			Gate& g = mlayer.gates[0];
			std::vector<Eigen::MatrixXf>& nodes = g.nodes;
			std::vector<Eigen::MatrixXf>& weights = g.weights;
			std::vector<Eigen::MatrixXf>& biases = g.biases;

			size_t filterVolume = layer.filter.width * layer.filter.height;
			size_t inputVolume = layer.input.width * layer.input.height;
			size_t outputVolume = layer.output.width * layer.output.height;
			
			for (size_t i = 0; i < m.count; ++i)
			{
				float* convBuffer = buffer[i].data();
				float* inputBuffer = inputs[i].data();

				//TODO:: make these loops efficient
				int ry = -layer.zeroPadding;
				for (size_t y = 0, buffidx = 0; y < layer.output.height; ++y, ry += layer.strideY)
				{
					int rx = -layer.zeroPadding;
					for (size_t x = 0; x < layer.output.width; ++buffidx, ++x, rx += layer.strideX)
					{
						for (int fy = 0, bufffidx = 0, fidx = 0; fy < layer.filter.height; ++fy)
						{
							for (int fx = 0; fx < layer.filter.width; ++bufffidx, ++fx, ++fidx)
							{
								for (size_t z = 0; z < layer.input.depth; ++z)
								{
									float val = 0;
									int locationX = rx + fx;
									int locationY = ry + fy;
									if (locationX >= 0 && locationY >= 0 && locationX < layer.input.width && locationY < layer.input.height)
									{
										size_t idx = locationX + locationY * layer.input.width;
										val = inputBuffer[idx * layer.input.depth + z];
									}
									convBuffer[buffidx * filterVolume * layer.input.depth + z * filterVolume + bufffidx] = val;
								}
							}
						}
					}
				}

				// https://eigen.tuxfamily.org/dox/TopicWritingEfficientProductExpression.html => refer to 4th down the graph "m1.noalias() = m4 + m2 * m3;"
				nodes[i] = biases[i];
				nodes[i].noalias() += weights[i] * buffer[i];

				nodes[i] = nodes[i].unaryExpr([](float val) { return tanh(val); });
			}
		}

		void Convolution(Agent, Model& m, ::TLK::Layer layer, size_t l, size_t count)
		{
			Layer& mlayer = m.mlayers[l];
			Gate& g = mlayer.gates[0];

			size_t filterVolume = layer.filter.width * layer.filter.height;
			size_t inputVolume = layer.input.width * layer.input.height;
			size_t outputVolume = layer.output.width * layer.output.height;

			for (size_t i = 0, index = m.count; i < count; ++i, ++index)
			{
				if (l != 0)
					mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.mlayers[l - 1].outputs[index].data(), layer.input.depth, inputVolume));
				else
				{
					m.inputs.push_back(Eigen::MatrixXf::Zero(layer.input.depth, inputVolume));
					mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.inputs[index].data(), layer.input.depth, inputVolume));
				}

				mlayer.buffer.push_back(Eigen::MatrixXf(filterVolume * layer.input.depth, outputVolume));
				g.nodes.push_back(Eigen::MatrixXf::Zero(layer.output.depth, outputVolume));
#ifdef TLK_DEBUGMODE
				g.biases.push_back(Eigen::MatrixXf::Ones(filterVolume * layer.input.depth, outputVolume));
				g.weights.push_back(Eigen::MatrixXf::Ones(layer.output.depth, filterVolume * layer.input.depth));
#else
				g.biases.push_back(Eigen::MatrixXf::Random(filterVolume * layer.input.depth, outputVolume));
				g.weights.push_back(Eigen::MatrixXf::Random(layer.output.depth, filterVolume * layer.input.depth));
#endif
				
				mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(g.nodes[index].data(), layer.output.depth, outputVolume));
				if (l == m.layers.size() - 1)
					m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(g.nodes[index].data(), layer.output.depth, outputVolume));
			}
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
	Layer::Impl LSTM{ 
		static_cast<void (*)(impl::Duplicate, Model&, Layer, size_t, size_t)>(impl::LSTM),
		static_cast<void (*)(impl::Compute, Model&, Layer, size_t)>(impl::LSTM),
		static_cast<void (*)(impl::Mutate, Model&, Layer, size_t, size_t)>(impl::LSTM),
		static_cast<void (*)(impl::Agent, Model&, Layer, size_t, size_t)>(impl::LSTM),
		static_cast<void (*)(impl::Remove, Model&, Layer, size_t, size_t)>(impl::LSTM),
		static_cast<void (*)(impl::Compile, Model&, Layer, size_t)>(impl::LSTM),
		static_cast<void (*)(impl::Recompile, Model&, Layer, size_t)>(impl::LSTM)
	};
	Layer::Impl Convolution{ 
		nullptr,
		static_cast<void (*)(impl::Compute, Model&, Layer, size_t)>(impl::Convolution),
		nullptr,
		static_cast<void (*)(impl::Agent, Model&, Layer, size_t, size_t)>(impl::Convolution),
		nullptr,
		static_cast<void (*)(impl::Compile, Model&, Layer, size_t)>(impl::Convolution),
		nullptr
	};
	Layer::Impl Pooling{
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		nullptr
	};
	Layer::Impl Flatten{
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		nullptr
	};

	void Model::Compile()
	{
		assert(!compiled);

		mlayers.reserve(layers.size());

		for (size_t i = 0; i < layers.size(); ++i)
		{
			mlayers.push_back(impl::Layer{});
			layers[i].impl.Compile({}, *this, layers[i], i);
		}

		compiled = true;
	}

	void Model::Append(Layer layer)
	{
		assert(!altered);
		assert(!compiled);
		assert(layers.size() == 0 || (layers.size() != 0 && layers.back().output == layer.input));

		layers.push_back(layer);
	}

	void Model::Agent(size_t count)
	{
		assert(!altered);
		assert(compiled);

		for (size_t i = 0; i < layers.size(); ++i)
			layers[i].impl.Agent({}, *this, layers[i], i, count);

		this->count += count;
	}

	void Model::Duplicate(size_t index)
	{
		assert(!altered);
		assert(compiled);

		for (size_t i = 0; i < layers.size(); ++i)
			layers[i].impl.Duplicate({}, *this, layers[i], i, index);

		++count;
	}

	void Model::Mutate(size_t index)
	{
		assert(compiled);

		for (size_t i = 0; i < layers.size(); ++i)
			layers[i].impl.Mutate({}, *this, layers[i], i, index);
	}

	void Model::RemoveAt(size_t index)
	{
		assert(compiled);
		assert(count > 0);

		altered = true;
		--count;

		for (size_t i = 0; i < layers.size(); ++i)
			layers[i].impl.Remove({}, *this, layers[i], i, index);
	}

	void Model::Recompile()
	{
		assert(altered);

		altered = false;

		inputs.clear();
		outputs.clear();
		for (size_t i = 0; i < layers.size(); ++i)
			layers[i].impl.Recompile({}, *this, layers[i], i);
	}

	void Model::Compute()
	{
		assert(!altered);
		assert(compiled);

		for (size_t i = 0; i < layers.size(); ++i)
			layers[i].impl.Compute({}, *this, layers[i], i);
	}
}