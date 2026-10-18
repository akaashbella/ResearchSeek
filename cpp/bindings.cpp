#include <pybind11/pybind11.h>

long diagnostic() {
    return __cplusplus;
}

PYBIND11_MODULE(_native, module) {
    module.doc() = "ResearchSeek native build diagnostics.";
    module.def("diagnostic", &diagnostic,
               "Return the C++ language standard used to compile this module.");
}
