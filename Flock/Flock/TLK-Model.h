#include <vector>
#include <Eigen/Dense>

#define TLK_DEBUGMODE

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
			void (* const Duplicate)(impl::Duplicate, Model&, Layer, size_t, size_t);
			void (* const Compute)(impl::Compute, Model&, Layer, size_t);
			void (* const Mutate)(impl::Mutate, Model&, Layer, size_t, size_t);
			void (* const Agent)(impl::Agent, Model&, Layer, size_t, size_t);
			void (* const Remove)(impl::Remove, Model&, Layer, size_t, size_t);
			void (* const Compile)(impl::Compile, Model&, Layer, size_t);
			void (* const Recompile)(impl::Recompile, Model&, Layer, size_t);

			Impl(
				void (*Duplicate)(impl::Duplicate, Model&, Layer, size_t, size_t),
				void (*Compute)(impl::Compute, Model&, Layer, size_t),
				void (*Mutate)(impl::Mutate, Model&, Layer, size_t, size_t),
				void (*Agent)(impl::Agent, Model&, Layer, size_t, size_t),
				void (*Remove)(impl::Remove, Model&, Layer, size_t, size_t),
				void (*Compile)(impl::Compile, Model&, Layer, size_t),
				void (*Recompile)(impl::Recompile, Model&, Layer, size_t)
			) : Duplicate(Duplicate), Compute(Compute), Mutate(Mutate), Agent(Agent), Remove(Remove), Compile(Compile), Recompile(Recompile) {}
		};

		const Tensor input;
		const Tensor output;
		const Tensor filter;
		const int strideX;
		const int strideY;
		const int zeroPadding;

		const Impl impl;

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
	};

	extern Layer::Impl Dense;
	extern Layer::Impl LSTM;
	extern Layer::Impl Convolution;

	struct Model
	{
		struct Impl;

		std::vector<Layer> layers;
		std::vector<impl::Layer> mlayers;

		std::vector<Eigen::MatrixXf> inputs;
		std::vector<Eigen::Map<Eigen::MatrixXf>> outputs;

		size_t count = 0;

		void Compile();

		void Duplicate(size_t index);
		void Mutate(size_t index);
		void Agent(size_t count);
		void Append(Layer layer);
		void RemoveAt(size_t i);
		void Recompile();

		void Compute();

	private:
		bool compiled = false;
		bool altered = false;
	};
}