#include <iostream>
#include <chrono>
#include <Eigen/Dense>

#include "TLK.h"

int main(int argc, char* argv[])
{
    std::srand((unsigned int)time(NULL));

    TLK::Model m{};
    m.Append(TLK::Layer(TLK::Dense, TLK::Tensor(1, 1), TLK::Tensor(1, 1)));
    m.Compile();
    m.Agent(1000);

    // Start Timer
    for (size_t i = 0; i < 100; i++)
    {
        auto t1 = std::chrono::high_resolution_clock::now();

        m.Compute();

        // End Timer
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        std::cout << ms_double.count() << "ms\n";
    }
}