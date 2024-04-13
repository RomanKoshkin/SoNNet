#!/bin/bash

echo "working directory: `pwd`"

if [ $1 = "openmp" ]; then
    echo "BUILDING FOR MULTI-CORE"
    g++ \
        -Ofast \
        -Wall \
        -mavx \
        -ftree-vectorize \
        -shared \
        -fPIC \
        -std=c++14 \
        -fopenmp \
        -lboost_system \
        $(python3 -m pybind11 --includes) \
        modules_pybind/main.cpp \
        -o modules_pybind/cpp_modules$(python3-config --extension-suffix)
fi

if [ $1 = "noopenmp" ]; then
    echo "BUILDING FOR SINGLE CORE"
    g++ \
        -Ofast \
        -Wall \
        -mavx \
        -ftree-vectorize \
        -shared \
        -fPIC \
        -std=c++14 \
        -lboost_system \
        $(python3 -m pybind11 --includes) \
        modules_pybind/main.cpp \
        -o modules_pybind/cpp_modules$(python3-config --extension-suffix)
fi

echo "c++ module built in: modules_pybind/"
