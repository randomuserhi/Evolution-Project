#include <vector>
#include <Eigen/Dense>

namespace TLK
{
	struct Tensor
	{
		size_t width;
		size_t height;
		size_t depth;

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
		struct Compile {};
		struct Compute {};
		struct Agent {};

		extern Compile compile;
		extern Compile compute;
		extern Agent append;

		struct Gate
		{
			std::vector<Eigen::MatrixXf> nodes;
			std::vector<Eigen::MatrixXf> weights;
			std::vector<Eigen::MatrixXf> biases;
		};

		struct Layer
		{
			std::vector<Gate> gates;
			std::vector<Eigen::Map<Eigen::MatrixXf>> inputs;
			std::vector<Eigen::Map<Eigen::MatrixXf>> outputs;
		};
	}

	struct Model;

	struct Layer
	{
		struct Impl
		{
			void (* const Compute)(impl::Compute, Model&, size_t);
			void (* const Agent)(impl::Agent, Model&, Layer, size_t, size_t);
			void (* const Compile)(impl::Compile, Model&, size_t);

			Impl(
				void (*Compute)(impl::Compute, Model&, size_t), 
				void (*Agent)(impl::Agent, Model&, Layer, size_t, size_t),
				void (*Compile)(impl::Compile, Model&, size_t)
			) : Compute(Compute), Agent(Agent), Compile(Compile) {}
		};

		const Tensor input;
		const Tensor output;

		const Impl impl;

		Layer(Impl type, Tensor input, Tensor output) : impl(type), input(input), output(output) {}
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

		int count = 0;

		void Compile();

		void Agent(size_t count);
		void Append(Layer layer);

		void Compute();

	private:
		bool compiled = false;
	};
}