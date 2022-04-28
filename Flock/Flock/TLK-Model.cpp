#include "TLK-Model.h"
#include <fstream>
#include <iostream>

//TODO:: comment implementation
//NOTE:: std::vector erase() seems to move Eigen::Map around causing problems so recompile function has to be used to remake all the maps.

namespace TLK
{
	void Model::Save(std::string fileName)
	{
		assert(!altered);
		assert(compiled);

		std::ofstream file;
		file.open(fileName, std::ios_base::binary);
		assert(file.is_open());

		file.write((char*)&count, sizeof count);

		for (size_t i = 0; i < layers.size(); ++i)
		{
			impl::Layer& layer = mlayers[i];
			for (size_t j = 0; j < layer.gates.size(); ++j)
			{
				impl::Gate& gate = layer.gates[j];
				for (size_t k = 0; k < count; ++k)
				{
					file.write((char*)gate.biases[k].data(), sizeof(float) * gate.biases[k].size());
					file.write((char*)gate.weights[k].data(), sizeof(float) * gate.weights[k].size());
				}
			}
		}

		file.close();
	}

	void Model::Load(std::string fileName)
	{
		assert(!altered);
		assert(compiled);

		std::ifstream file;
		file.open(fileName, std::ios_base::binary);
		
		size_t total;
		file.read((char*)&total, sizeof total);
		if (count < total) Agent(total - count);

		for (size_t i = 0; i < layers.size(); ++i)
		{
			impl::Layer& layer = mlayers[i];
			for (size_t j = 0; j < layer.gates.size(); ++j)
			{
				impl::Gate& gate = layer.gates[j];
				for (size_t k = 0; k < count; ++k)
				{
					file.read((char*)gate.biases[k].data(), sizeof(float) * gate.biases[k].size());
					file.read((char*)gate.weights[k].data(), sizeof(float) * gate.weights[k].size());
				}
			}
		}

		file.close();
	}

	void Model::Copy(size_t dst, size_t src)
	{
		assert(!altered);
		assert(compiled);

		for (size_t i = 0; i < layers.size(); ++i)
		{
			impl::Layer& layer = mlayers[i];
			for (size_t j = 0; j < layer.gates.size(); ++j)
			{
				impl::Gate& gate = layer.gates[j];
				memcpy(gate.biases[dst].data(), gate.biases[src].data(), sizeof(float) * gate.biases[src].size());
				memcpy(gate.weights[dst].data(), gate.weights[src].data(), sizeof(float) * gate.weights[src].size());
			}
		}
	}
}