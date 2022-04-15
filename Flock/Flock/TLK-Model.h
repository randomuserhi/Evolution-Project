#include <vector>
#include <Eigen/Dense>

namespace TLK
{
	typedef std::vector<Eigen::MatrixXf> MatLayer;

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

	struct Model;
	struct Layer
	{
		struct Impl
		{
			void (* const Append)(Model&, Layer, size_t, size_t);
			void (* const Compute)(Model&, Layer, size_t);

			Impl(void (* const compile)(Model&, Layer, size_t, size_t), void (* const compute)(Model&, Layer, size_t)) : Append(compile), Compute(compute) {};
		};

		const Tensor input;
		const Tensor output;
		const Impl methods;

		Layer(Impl type, Tensor input, Tensor output) : methods(type), input(input), output(output) {};
	};

	const extern Layer::Impl Dense;
	const extern Layer::Impl LSTM;
	const extern Layer::Impl Pooling;
	const extern Layer::Impl Convolution;

	struct Model
	{
		std::vector<Layer> layers;

		MatLayer* mlayers;
		MatLayer* weights;
		MatLayer* biases;

		~Model()
		{
			delete[] mlayers;
			delete[] weights;
			delete[] biases;
		}

		void Compile();
		void Append(Layer layer);
		void Agent(size_t count = 1);
		void Compute();
	};
}