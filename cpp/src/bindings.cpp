#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "morphosml/vector.hpp"
#include "morphosml/matrix.hpp"
#include "morphosml/knn.hpp"
#include "morphosml/metrics.hpp"

namespace py = pybind11;

morphosml::Matrix numpy_to_matrix(py::array_t<double> arr) {

    py::buffer_info buf =
        const_cast<py::array_t<double>&>(arr).request();

    if (buf.ndim != 2) {
        throw std::runtime_error("The array needs to be 2D.");
    }

    morphosml::Matrix result(buf.shape[0], buf.shape[1]);

    double* ptr = static_cast<double*>(buf.ptr);

    for (size_t i = 0; i < static_cast<size_t>(buf.shape[0]); ++i) {

        for (size_t j = 0; j < static_cast<size_t>(buf.shape[1]); ++j) {

            result(i, j) =
                ptr[i * buf.shape[1] + j];
        }
    }

    return result;
}