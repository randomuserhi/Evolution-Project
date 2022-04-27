#include "TLK-Model.h"
#include <iostream>

namespace TLK
{
	namespace impl
	{
		void Pooling(Compile, Model& m, ::TLK::Layer layer, size_t l)
		{
			assert(layer.input.width > 0 && layer.input.height > 0 && layer.input.depth > 0);
			assert(layer.output.width > 0 && layer.output.height > 0 && layer.output.depth > 0);
			assert(layer.input.depth == layer.output.depth);
			assert(layer.filter.width > 0 && layer.filter.height > 0 && layer.filter.depth > 0);
			assert(layer.strideX > 0 && layer.strideY > 0 && layer.zeroPadding >= 0);
		}

		void Pooling(Agent, Model& m, ::TLK::Layer layer, size_t l, size_t count)
		{
			Layer& mlayer = m.mlayers[l];

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

				mlayer.buffer.push_back(Eigen::MatrixXf(layer.output.depth, outputVolume));

				mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.buffer[index].data(), layer.output.depth, outputVolume));
				if (l == m.layers.size() - 1)
					m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.buffer[index].data(), layer.output.depth, outputVolume));
			}
		}

		void Pooling(Recompile, Model& m, ::TLK::Layer layer, size_t l)
		{
			Layer& mlayer = m.mlayers[l];

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

				mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.buffer[i].data(), layer.output.depth, outputVolume));
				if (l == m.layers.size() - 1)
					m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.buffer[i].data(), layer.output.depth, outputVolume));
			}
		}

		void Pooling(Duplicate, Model& m, ::TLK::Layer layer, size_t l, size_t index)
		{
			Layer& mlayer = m.mlayers[l];

			size_t inputVolume = layer.input.width * layer.input.height;
			size_t outputVolume = layer.output.width * layer.output.height;

			if (l != 0)
				mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.mlayers[l - 1].outputs[m.count].data(), layer.input.depth, inputVolume));
			else
			{
				m.inputs.push_back(Eigen::MatrixXf::Zero(layer.input.depth, inputVolume));
				mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.inputs[m.count].data(), layer.input.depth, inputVolume));
			}

			mlayer.buffer.push_back(Eigen::MatrixXf(layer.output.depth, outputVolume));

			mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.buffer[m.count].data(), layer.output.depth, outputVolume));
			if (l == m.layers.size() - 1)
				m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.buffer[m.count].data(), layer.output.depth, outputVolume));
		}

		void Pooling(Compute, Model& m, ::TLK::Layer layer, size_t l)
		{
			Layer& mlayer = m.mlayers[l];
			std::vector<Eigen::Map<Eigen::MatrixXf>>& inputs = mlayer.inputs;
			std::vector<Eigen::MatrixXf>& buffer = mlayer.buffer;

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
						for (size_t z = 0; z < layer.input.depth; ++z)
						{
							float val = 0;
							int locationX = rx;
							int locationY = ry;
							if (locationX >= 0 && locationY >= 0 && locationX < layer.input.width && locationY < layer.input.height)
							{
								size_t idx = locationX + locationY * layer.input.width;
								val = inputBuffer[idx * layer.input.depth + z];
							}

							for (int fy = 0, fidx = 0; fy < layer.filter.height; ++fy)
							{
								for (int fx = 0; fx < layer.filter.width; ++fx, ++fidx)
								{
									float v = 0;
									int locationX = rx + fx;
									int locationY = ry + fy;
									if (locationX >= 0 && locationY >= 0 && locationX < layer.input.width && locationY < layer.input.height)
									{
										size_t idx = locationX + locationY * layer.input.width;
										v = inputBuffer[idx * layer.input.depth + z];
									}

									if (v > val)
									{
										val = v;
									}
								}
							}

							convBuffer[buffidx] = val;
						}
					}
				}
			}
		}

		void Pooling(Remove, Model& m, ::TLK::Layer layer, size_t l, size_t i)
		{
			Layer& mlayer = m.mlayers[l];

			mlayer.buffer.erase(mlayer.buffer.begin() + i);
		}

		void Pooling(Mutate, Model& m, ::TLK::Layer layer, size_t l, size_t i)
		{
		}
	}

	Layer::Impl Pooling{
		static_cast<void (*)(impl::Duplicate, Model&, Layer, size_t, size_t)>(impl::Pooling),
		static_cast<void (*)(impl::Compute, Model&, Layer, size_t)>(impl::Pooling),
		static_cast<void (*)(impl::Mutate, Model&, Layer, size_t, size_t)>(impl::Pooling),
		static_cast<void (*)(impl::Agent, Model&, Layer, size_t, size_t)>(impl::Pooling),
		static_cast<void (*)(impl::Remove, Model&, Layer, size_t, size_t)>(impl::Pooling),
		static_cast<void (*)(impl::Compile, Model&, Layer, size_t)>(impl::Pooling),
		static_cast<void (*)(impl::Recompile, Model&, Layer, size_t)>(impl::Pooling)
	};
}