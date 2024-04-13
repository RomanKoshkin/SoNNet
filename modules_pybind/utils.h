#include <boost/random.hpp>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <vector>

using namespace std;

struct Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<float> duration;

    Timer() { start = std::chrono::high_resolution_clock::now(); }

    /* when the function where this object is created returns,
      this object must be destroyed, hence this destructor is called */
    ~Timer() {
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        float ms = duration.count() * 1000.0f;
        // std::cout << "Elapsed (c++ timer): " << ms << " ms." << std::endl;
    }
};