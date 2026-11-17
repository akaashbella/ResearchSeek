cmake_minimum_required(VERSION 3.18)
project(ResearchSeek LANGUAGES CXX)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_native cpp/bindings.cpp cpp/vector_search.cpp)
set_target_properties(_native PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
install(TARGETS _native LIBRARY DESTINATION researchseek)

option(RESEARCHSEEK_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(RESEARCHSEEK_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(FATAL_ERROR "Sanitizers require Clang or GCC")
    endif()
    target_compile_options(_native PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(_native PRIVATE -fsanitize=address,undefined)
endif()
