#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "vector_search.hpp"

#include <algorithm>
#include <cstdint>
#include <string>

namespace py = pybind11;

namespace {

template <typename T>
void require_layout(const py::array& array, const char* name) {
    if (!array.dtype().equal(py::dtype::of<T>())) {
        throw py::type_error(std::string(name) + " has an incorrect dtype; use native float32 vectors and int64 IDs");
    }
    if (!(array.flags() & py::array::c_style)) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
    if (reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) != 0) {
        throw py::value_error(std::string(name) + " must be aligned");
    }
}

py::tuple top_k(py::array matrix, py::array query, py::array ids, py::object k_object) {
    require_layout<float>(matrix, "matrix");
    require_layout<float>(query, "query");
    require_layout<std::int64_t>(ids, "chunk_ids");
    if (matrix.ndim() != 2 || query.ndim() != 1 || ids.ndim() != 1) {
        throw py::value_error("Expected matrix (N, d), query (d,), and chunk_ids (N,)");
    }
    if (matrix.shape(1) < 1 || query.shape(0) != matrix.shape(1) || ids.shape(0) != matrix.shape(0)) {
        throw py::value_error("Input dimensions/lengths must match and d must be positive");
    }
    if (PyBool_Check(k_object.ptr()) || !PyIndex_Check(k_object.ptr())) {
        throw py::type_error("k must be a positive integer");
    }
    py::object integer = py::reinterpret_steal<py::object>(PyNumber_Index(k_object.ptr()));
    if (!integer) {
        throw py::error_already_set();
    }
    int overflow = 0;
    const auto requested_k = PyLong_AsLongLongAndOverflow(integer.ptr(), &overflow);
    if (PyErr_Occurred()) {
        throw py::error_already_set();
    }
    if (overflow < 0 || (overflow == 0 && requested_k < 1)) {
        throw py::value_error("k must be positive");
    }
    const auto rows = static_cast<std::size_t>(matrix.shape(0));
    const auto capacity = std::max<std::size_t>(rows, 1);
    const auto k = overflow > 0 || requested_k > static_cast<long long>(capacity)
                       ? capacity : static_cast<std::size_t>(requested_k);
    const auto dimension = static_cast<std::size_t>(matrix.shape(1));
    const auto* vectors = static_cast<const float*>(matrix.data());
    const auto* query_data = static_cast<const float*>(query.data());
    const auto* chunk_ids = static_cast<const std::int64_t*>(ids.data());
    researchseek::validate_inputs(vectors, query_data, chunk_ids, rows, dimension, k);
    std::vector<researchseek::Match> matches;
    {
        py::gil_scoped_release release;
        matches = researchseek::cosine_top_k(vectors, query_data, chunk_ids, rows, dimension, k);
    }
    py::array_t<std::int64_t> result_ids(matches.size());
    py::array_t<double> scores(matches.size());
    for (std::size_t index = 0; index < matches.size(); ++index) {
        result_ids.mutable_data()[index] = matches[index].chunk_id;
        scores.mutable_data()[index] = matches[index].score;
    }
    return py::make_tuple(result_ids, scores);
}

}  // namespace

long diagnostic() {
    return __cplusplus;
}

PYBIND11_MODULE(_native, module) {
    module.doc() = "ResearchSeek exact cosine retrieval and build diagnostics.";
    module.def("diagnostic", &diagnostic,
               "Return the C++ language standard used to compile this module.");
    module.def("top_k", &top_k, py::arg("matrix").noconvert(), py::arg("query").noconvert(),
               py::arg("chunk_ids").noconvert(), py::arg("k"),
               "Exact cosine top-k; inputs are borrowed without casts/copies and must not be mutated during the call.");
}
