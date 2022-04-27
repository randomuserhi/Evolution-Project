#include "TLK-Model.h"
#include <iostream>

namespace TLK
{
	namespace impl
	{
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
				g.biases.push_back(Eigen::MatrixXf::Ones(layer.output.depth, outputVolume));
				g.weights.push_back(Eigen::MatrixXf::Ones(layer.output.depth, filterVolume * layer.input.depth));
#else
				g.biases.push_back(Eigen::MatrixXf::Random(layer.output.depth, outputVolume));
				g.weights.push_back(Eigen::MatrixXf::Random(layer.output.depth, filterVolume * layer.input.depth));
#endif

				mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(g.nodes[index].data(), layer.output.depth, outputVolume));
				if (l == m.layers.size() - 1)
					m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(g.nodes[index].data(), layer.output.depth, outputVolume));
			}
		}

		void Convolution(Recompile, Model& m, ::TLK::Layer layer, size_t l)
		{
			Layer& mlayer = m.mlayers[l];
			Gate& g = mlayer.gates[0];

			size_t filterVolume = layer.filter.width * layer.filter.height;
			size_t inputVolume = layer.input.width * layer.input.height;
			size_t outputVolume = layer.output.width * layer.output.height;

			for (size_t i = 0; i < m.count; ++i)
			{
				if (l != 0)
					mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.mlayers[l - 1].outputs[i].data(), layer.input.depth, inputVolume));
				else
				{
					m.inputs.push_back(Eigen::MatrixXf::Zero(layer.input.depth, inputVolume));
					mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.inputs[i].data(), layer.input.depth, inputVolume));
				}

				mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(g.nodes[i].data(), layer.output.depth, outputVolume));
				if (l == m.layers.size() - 1)
					m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(g.nodes[i].data(), layer.output.depth, outputVolume));
			}
		}

		void Convolution(Duplicate, Model& m, ::TLK::Layer layer, size_t l, size_t index)
		{
			Layer& mlayer = m.mlayers[l];
			Gate& g = mlayer.gates[0];

			size_t filterVolume = layer.filter.width * layer.filter.height;
			size_t inputVolume = layer.input.width * layer.input.height;
			size_t outputVolume = layer.output.width * layer.output.height;

			if (l != 0)
				mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.mlayers[l - 1].outputs[m.count].data(), layer.input.depth, inputVolume));
			else
			{
				m.inputs.push_back(Eigen::MatrixXf::Zero(layer.input.depth, inputVolume));
				mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.inputs[m.count].data(), layer.input.depth, inputVolume));
			}

			mlayer.buffer.push_back(Eigen::MatrixXf(filterVolume * layer.input.depth, outputVolume));
			g.nodes.push_back(Eigen::MatrixXf::Zero(layer.output.depth, outputVolume));

			g.biases.push_back(Eigen::MatrixXf(layer.output.depth, outputVolume));
			g.weights.push_back(Eigen::MatrixXf(layer.output.depth, filterVolume * layer.input.depth));
			memcpy(g.weights[m.count].data(), g.weights[index].data(), sizeof(float) * g.weights[m.count].size());
			memcpy(g.biases[m.count].data(), g.biases[index].data(), sizeof(float) * g.biases[m.count].size());

			mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(g.nodes[m.count].data(), layer.output.depth, outputVolume));
			if (l == m.layers.size() - 1)
				m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(g.nodes[m.count].data(), layer.output.depth, outputVolume));
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

		void Convolution(Remove, Model& m, ::TLK::Layer layer, size_t l, size_t i)
		{
			Layer& mlayer = m.mlayers[l];
			Gate& g = mlayer.gates[0];

			mlayer.buffer.erase(mlayer.buffer.begin() + i);
			g.nodes.erase(g.nodes.begin() + i);
			g.weights.erase(g.weights.begin() + i);
			g.biases.erase(g.biases.begin() + i);
		}

		void Convolution(Mutate, Model& m, ::TLK::Layer layer, size_t l, size_t i)
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

	Layer::Impl Convolution{
		static_cast<void (*)(impl::Duplicate, Model&, Layer, size_t, size_t)>(impl::Convolution),
		static_cast<void (*)(impl::Compute, Model&, Layer, size_t)>(impl::Convolution),
		static_cast<void (*)(impl::Mutate, Model&, Layer, size_t, size_t)>(impl::Convolution),
		static_cast<void (*)(impl::Agent, Model&, Layer, size_t, size_t)>(impl::Convolution),
		static_cast<void (*)(impl::Remove, Model&, Layer, size_t, size_t)>(impl::Convolution),
		static_cast<void (*)(impl::Compile, Model&, Layer, size_t)>(impl::Convolution),
		static_cast<void (*)(impl::Recompile, Model&, Layer, size_t)>(impl::Convolution)
	};
}