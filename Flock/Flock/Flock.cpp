#include <iostream>
#include <chrono>
#include <Eigen/Dense>

#include "TLK.h"

int main(int argc, char* argv[])
{
    std::srand((unsigned int)time(NULL));

    TLK::Model m{};
    m.Append(TLK::Layer(TLK::Pooling, TLK::Tensor(100, 100, 1), TLK::Tensor(2, 2), 2, 2));
    //m.Append(TLK::Layer(TLK::Flatten, TLK::Tensor(2, 2, 2), TLK::Tensor(8, 1)));
    m.Compile();
    m.Agent(1);

    m.Duplicate(0);
    m.Mutate(1);

    m.RemoveAt(0);
    m.Recompile();

    // Start Timer
    for (size_t i = 0; i < 1; ++i)
    {
        auto t1 = std::chrono::high_resolution_clock::now();

        m.Compute();

        // End Timer
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        std::cout << ms_double.count() << "ms\n";
    }

    /*for (size_t i = 0; i < m.count; ++i)
    {
        std::cout << m.outputs[i] << std::endl;
    }*/
}