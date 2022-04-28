#include <vector>
#include <Eigen/Dense>

//#define TLK_DEBUGMODE

namespace TLK
{
	struct Tensor
	{
		size_t width = 0;
		size_t height = 0;
		size_t depth = 0;

		Tensor() {};
		Tensor(size_t width, size_t height) : width(width), height(height), depth(1) {}
		Tensor(size_t width, size_t height, size_t depth) : width(width), height(height), depth(depth) {}

		friend bool operator==(const Tensor& a, const Tensor& b);
		friend bool operator!=(const Tensor& a, const Tensor& b);
	};

	inline bool operator== (const Tensor& a, const Tensor& b)
	{
		return a.width == b.width && a.height == b.height && a.depth == b.depth;
	}
	inline bool operator!= (const Tensor& a, const Tensor& b)
	{
		return a.width != b.width || a.height != b.height || a.depth != b.depth;
	}

	namespace impl
	{
		struct Reset {};
		struct Duplicate{};
		struct Compile {};
		struct Mutate {};
		struct Compute {};
		struct Agent {};
		struct Remove {};
		struct Recompile {};

		struct Gate
		{
			std::vector<Eigen::MatrixXf> nodes;
			std::vector<Eigen::MatrixXf> biases;
			std::vector<Eigen::MatrixXf> weights;
		};

		struct Layer
		{
			std::vector<Gate> gates;
			std::vector<Eigen::MatrixXf> state;
			std::vector<Eigen::MatrixXf> buffer;
			std::vector<Eigen::Map<Eigen::MatrixXf>> inputs;
			std::vector<Eigen::Map<Eigen::MatrixXf>> outputs;
		};
	}

	struct Model;

	struct Layer
	{
		struct Impl
		{
			void (* const Reset)(impl::Reset, Model&, Layer, size_t, size_t);
			void (* const Duplicate)(impl::Duplicate, Model&, Layer, size_t, size_t);
			void (* const Compute)(impl::Compute, Model&, Layer, size_t);
			void (* const Mutate)(impl::Mutate, Model&, Layer, size_t, size_t);
			void (* const Agent)(impl::Agent, Model&, Layer, size_t, size_t);
			void (* const Remove)(impl::Remove, Model&, Layer, size_t, size_t);
			void (* const Compile)(impl::Compile, Model&, Layer, size_t);
			void (* const Recompile)(impl::Recompile, Model&, Layer, size_t);

			Impl(
				void (*Reset)(impl::Reset, Model&, Layer, size_t, size_t),
				void (*Duplicate)(impl::Duplicate, Model&, Layer, size_t, size_t),
				void (*Compute)(impl::Compute, Model&, Layer, size_t),
				void (*Mutate)(impl::Mutate, Model&, Layer, size_t, size_t),
				void (*Agent)(impl::Agent, Model&, Layer, size_t, size_t),
				void (*Remove)(impl::Remove, Model&, Layer, size_t, size_t),
				void (*Compile)(impl::Compile, Model&, Layer, size_t),
				void (*Recompile)(impl::Recompile, Model&, Layer, size_t)
			) : Reset(Reset), Duplicate(Duplicate), Compute(Compute), Mutate(Mutate), Agent(Agent), Remove(Remove), Compile(Compile), Recompile(Recompile) {}
		};

		const Tensor input;
		const Tensor output;
		const Tensor filter;
		const int strideX;
		const int strideY;
		const int zeroPadding;

		const Impl impl;

		//Add overload dispatch tags to make these less ambiguous
		Layer(Impl type, Tensor input, Tensor output) : 
			impl(type), input(input), output(output),
			filter(), strideX(0), strideY(0), zeroPadding(0)
		{}
		Layer(Impl type, Tensor input, Tensor filter, size_t numFilters, int strideX = 1, int strideY = 1, int zeroPadding = 0) :
			impl(type), input(input), output(
				std::ceil((input.width + zeroPadding * 2 - (filter.width - 1)) / (float)strideX),
				std::ceil((input.height + zeroPadding * 2 - (filter.height - 1)) / (float)strideY),
				numFilters),
			filter(filter.width, filter.height, input.depth), strideX(strideX), strideY(strideY), zeroPadding(zeroPadding)
		{}
		Layer(Impl type, Tensor input, Tensor filter, int strideX, int strideY, int zeroPadding = 0) :
			impl(type), input(input), output(
				std::ceil((input.width + zeroPadding * 2 - (filter.width - 1)) / (float)strideX),
				std::ceil((input.height + zeroPadding * 2 - (filter.height - 1)) / (float)strideY),
				input.depth),
			filter(filter.width, filter.height, input.depth), strideX(strideX), strideY(strideY), zeroPadding(zeroPadding)
		{}
	};

	extern Layer::Impl Dense;
	extern Layer::Impl LSTM;
	extern Layer::Impl Convolution;
	extern Layer::Impl Flatten;
	extern Layer::Impl Pooling;

	struct Model
	{
		struct Impl;

		std::vector<Layer> layers;
		std::vector<impl::Layer> mlayers;

		std::vector<Eigen::MatrixXf> inputs;
		std::vector<Eigen::Map<Eigen::MatrixXf>> outputs;

		size_t count = 0;

		inline void Compile()
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

		inline void Append(Layer layer)
		{
			assert(!altered);
			assert(!compiled);
			assert(layers.size() == 0 || (layers.size() != 0 && layers.back().output == layer.input));

			layers.push_back(layer);
		}

		inline void Agent(size_t count)
		{
			assert(!altered);
			assert(compiled);

			for (size_t i = 0; i < layers.size(); ++i)
				layers[i].impl.Agent({}, *this, layers[i], i, count);

			this->count += count;
		}

		inline void Reset(size_t index)
		{
			assert(!altered);
			assert(compiled);

			for (size_t i = 0; i < layers.size(); ++i)
				layers[i].impl.Reset({}, *this, layers[i], i, index);
		}

		inline void Duplicate(size_t index)
		{
			assert(!altered);
			assert(compiled);

			for (size_t i = 0; i < layers.size(); ++i)
				layers[i].impl.Duplicate({}, *this, layers[i], i, index);

			++count;
		}

		inline void Mutate(size_t index)
		{
			assert(compiled);

			for (size_t i = 0; i < layers.size(); ++i)
				layers[i].impl.Mutate({}, *this, layers[i], i, index);
		}

		inline void RemoveAt(size_t index)
		{
			assert(compiled);
			assert(count > 0);

			altered = true;
			--count;

			for (size_t i = 0; i < layers.size(); ++i)
				layers[i].impl.Remove({}, *this, layers[i], i, index);
		}

		inline void Recompile()
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

		inline void Compute()
		{
			assert(!altered);
			assert(compiled);

			for (size_t i = 0; i < layers.size(); ++i)
				layers[i].impl.Compute({}, *this, layers[i], i);
		}

		void Copy(size_t dst, size_t src);

		void Save(std::string fileName);
		void Load(std::string fileName);

	private:
		bool compiled = false;
		bool altered = false;
	};
}