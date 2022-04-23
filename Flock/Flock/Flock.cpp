#include <iostream>
#include <chrono>
#include <Eigen/Dense>

#include "TLK.h"

int main(int argc, char* argv[])
{
    TLK::Model m{};
    m.Append(TLK::Layer(TLK::Dense, TLK::Tensor{ 3, 0, 0 }, TLK::Tensor{ 2, 0, 0 }));
    m.Append(TLK::Layer(TLK::Dense, TLK::Tensor{ 2, 0, 0 }, TLK::Tensor{ 1, 0, 0 }));
    m.Compile();
    m.Agent(1);

    std::cout << m.count << std::endl;

    // Start Timer
    //for (size_t i = 0; i < 20; i++)
    {
        auto t1 = std::chrono::high_resolution_clock::now();

        m.Compute();

        // End Timer
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        std::cout << ms_double.count() << "ms\n";
    }

    std::cout << m.outputs[0] << std::endl;
}