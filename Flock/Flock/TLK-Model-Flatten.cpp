#include "TLK-Model.h"
#include <iostream>

namespace TLK
{
	namespace impl
	{
		void Flatten(Compile, Model& m, ::TLK::Layer layer, size_t l)
		{
			assert(layer.input.width > 0 && layer.input.height > 0 && layer.input.depth > 0);
			assert(layer.output.width == layer.input.width * layer.input.height * layer.input.depth);
		}

		void Flatten(Agent, Model& m, ::TLK::Layer layer, size_t l, size_t count)
		{
			Layer& mlayer = m.mlayers[l];

			size_t inputVolume = layer.input.width * layer.input.height;

			for (size_t i = 0, index = m.count; i < count; ++i, ++index)
			{
				if (l != 0)
					mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.mlayers[l - 1].outputs[index].data(), layer.input.depth, inputVolume));
				else
				{
					m.inputs.push_back(Eigen::MatrixXf::Zero(layer.input.depth, inputVolume));
					mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.inputs[index].data(), layer.input.depth, inputVolume));
				}

				mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.inputs[index].data(), 1, layer.output.width));
				if (l == m.layers.size() - 1)
					m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.inputs[index].data(), 1, layer.output.width));
			}
		}

		void Flatten(Recompile, Model& m, ::TLK::Layer layer, size_t l)
		{
			Layer& mlayer = m.mlayers[l];

			size_t inputVolume = layer.input.width * layer.input.height;

			for (size_t i = 0; i < m.count; ++i)
			{
				if (l != 0)
					mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.mlayers[l - 1].outputs[i].data(), layer.input.depth, inputVolume));
				else
				{
					m.inputs.push_back(Eigen::MatrixXf::Zero(layer.input.depth, inputVolume));
					mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.inputs[i].data(), layer.input.depth, inputVolume));
				}

				mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.inputs[i].data(), 1, layer.output.width));
				if (l == m.layers.size() - 1)
					m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.inputs[i].data(), 1, layer.output.width));
			}
		}

		void Flatten(Reset, Model& m, ::TLK::Layer layer, size_t l, size_t index)
		{
		}

		void Flatten(Duplicate, Model& m, ::TLK::Layer layer, size_t l, size_t index)
		{
			Layer& mlayer = m.mlayers[l];

			size_t inputVolume = layer.input.width * layer.input.height;

			if (l != 0)
				mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.mlayers[l - 1].outputs[m.count].data(), layer.input.depth, inputVolume));
			else
			{
				m.inputs.push_back(Eigen::MatrixXf::Zero(layer.input.depth, inputVolume));
				mlayer.inputs.push_back(Eigen::Map<Eigen::MatrixXf>(m.inputs[m.count].data(), layer.input.depth, inputVolume));
			}

			mlayer.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.inputs[m.count].data(), 1, layer.output.width));
			if (l == m.layers.size() - 1)
				m.outputs.push_back(Eigen::Map<Eigen::MatrixXf>(mlayer.inputs[m.count].data(), 1, layer.output.width));
		}

		//TODO:: maybe abstract the layer information and layers for computation.
		//		 basically seperate the layers used in compilation and the impl:: functions
		//		 so that the compilation process can just reshape the previous
		//       layers output to flatten, as opposed to having a layer with
		//       an empty compute function call
		void Flatten(Compute, Model& m, ::TLK::Layer layer, size_t l)
		{
		}

		void Flatten(Remove, Model& m, ::TLK::Layer layer, size_t l, size_t index)
		{
		}

		void Flatten(Mutate, Model& m, ::TLK::Layer layer, size_t l, size_t i)
		{
		}
	}

	Layer::Impl Flatten{
		static_cast<void (*)(impl::Reset, Model&, Layer, size_t, size_t)>(impl::Flatten),
		static_cast<void (*)(impl::Duplicate, Model&, Layer, size_t, size_t)>(impl::Flatten),
		static_cast<void (*)(impl::Compute, Model&, Layer, size_t)>(impl::Flatten),
		static_cast<void (*)(impl::Mutate, Model&, Layer, size_t, size_t)>(impl::Flatten),
		static_cast<void (*)(impl::Agent, Model&, Layer, size_t, size_t)>(impl::Flatten),
		static_cast<void (*)(impl::Remove, Model&, Layer, size_t, size_t)>(impl::Flatten),
		static_cast<void (*)(impl::Compile, Model&, Layer, size_t)>(impl::Flatten),
		static_cast<void (*)(impl::Recompile, Model&, Layer, size_t)>(impl::Flatten)
	};
}