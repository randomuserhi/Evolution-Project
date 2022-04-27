#include "TLK-Model.h"
#include <iostream>

//TODO:: comment implementation
//
//Convolution layers are too slow performance wise to use in my application! D:

//NOTE:: std::vector erase() seems to move Eigen::Map around causing problems so recompile function has to be used to remake all the maps.

namespace TLK
{
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
		{
			mlayers[i].inputs.clear();
			mlayers[i].outputs.clear();
			layers[i].impl.Recompile({}, *this, layers[i], i);
		}
	}

	void Model::Compute()
	{
		assert(!altered);
		assert(compiled);

		for (size_t i = 0; i < layers.size(); ++i)
			layers[i].impl.Compute({}, *this, layers[i], i);
	}
}