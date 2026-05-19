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

PYBIND11_MODULE(_core, m) {
    m.doc() = "MorphosML C++ core library";
    
    // Vector bindings
    py::class_<morphosml::Vector>(m, "Vector")
        .def(py::init<>())
        .def(py::init<size_t>())
        .def(py::init<std::vector<double>>())
        .def("size", &morphosml::Vector::size)
        .def("__getitem__", [](morphosml::Vector& v, size_t i) { return v[i]; })
        .def("__setitem__", [](morphosml::Vector& v, size_t i, double val) { v[i] = val; })
        .def("dot", &morphosml::Vector::dot)
        .def("norm", &morphosml::Vector::norm)
        .def("normalized", &morphosml::Vector::normalized)
        .def("to_list", &morphosml::Vector::to_std);
    
    // Matrix bindings
    py::class_<morphosml::Matrix>(m, "Matrix")
        .def(py::init<>())
        .def(py::init<size_t, size_t>())
        .def(py::init<std::vector<std::vector<double>>>())
        .def("rows", &morphosml::Matrix::rows)
        .def("cols", &morphosml::Matrix::cols)
        .def("__call__", [](morphosml::Matrix& mat, size_t i, size_t j) -> double& { 
            return mat(i, j); 
        })
        .def("transpose", &morphosml::Matrix::transpose)
        .def("__mul__", [](const morphosml::Matrix& a, const morphosml::Matrix& b) {
            return a * b;
        })
        .def_static("identity", &morphosml::Matrix::identity)
        .def_static("zeros", &morphosml::Matrix::zeros)
        .def_static("random", &morphosml::Matrix::random)
        .def("to_list", &morphosml::Matrix::to_std);
    
    // KNN bindings
    py::class_<morphosml::KNN>(m, "KNN")
        .def(py::init<int>(), py::arg("k") = 3)
        .def("fit", [](morphosml::KNN& knn, py::array_t<double> X, std::vector<int> y) {
            knn.fit(numpy_to_matrix(X), y);
        })
        .def("predict", [](morphosml::KNN& knn, py::array_t<double> X) {
            return knn.predict(numpy_to_matrix(X));
        })
        .def_property_readonly("k", &morphosml::KNN::get_k);
    
    // Metrics bindings
    m.def("accuracy", &morphosml::accuracy);
    m.def("mse", &morphosml::mse);
}
