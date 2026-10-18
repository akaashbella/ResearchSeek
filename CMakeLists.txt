cmake_minimum_required(VERSION 3.18)
project(ResearchSeek LANGUAGES CXX)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_native cpp/bindings.cpp)
set_target_properties(_native PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
install(TARGETS _native LIBRARY DESTINATION researchseek)
