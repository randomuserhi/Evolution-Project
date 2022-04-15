#include <iostream>
#include <chrono>
#include <Eigen/Dense>

#include "TLK.h"

int main(int argc, char* argv[])
{
    TLK::Model m{};
    m.Append(TLK::Layer(TLK::Dense, TLK::Tensor{ 3, 0, 0 }, TLK::Tensor{ 1, 0, 0 }));
    m.Append(TLK::Layer(TLK::Dense, TLK::Tensor{ 1, 0, 0 }, TLK::Tensor{ 1, 0, 0 }));
    m.Compile();
    m.Agent(1000);

    //m.mlayers[0][0] << 1, 1, 1;

    // Start Timer
    auto t1 = std::chrono::high_resolution_clock::now();

    m.Compute();

    // End Timer
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";
    
    //std::cout << m.mlayers[1][0] << std::endl;
    //std::cout << m.mlayers[2][0] << std::endl;
}