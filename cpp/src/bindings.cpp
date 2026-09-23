#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>

#include "morphosml/vector.hpp"
#include "morphosml/matrix.hpp"
#include "morphosml/tensor_view.hpp"
#include "morphosml/knn.hpp"
#include "morphosml/linear_regression.hpp"
#include "morphosml/logistic_regression.hpp"
#include "morphosml/metrics.hpp"
#include "morphosml/data/cursor.hpp"
#include "morphosml/data/sampler.hpp"
#include "morphosml/data/mmap_buffer.hpp"

namespace py = pybind11;

// HPC fast contiguous copy from NumPy to flat Matrix
morphosml::Matrix numpy_to_matrix(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info buf = arr.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("The input NumPy array must be 2-dimensional.");
    }

    const size_t rows = static_cast<size_t>(buf.shape[0]);
    const size_t cols = static_cast<size_t>(buf.shape[1]);

    std::vector<double> flat_data(rows * cols);
    std::memcpy(flat_data.data(), buf.ptr, rows * cols * sizeof(double));

    return morphosml::Matrix(rows, cols, std::move(flat_data));
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "MorphosML C++ core library with HPC linear algebra and models";

    // Global Seed & Reproducibility System
    m.def("set_seed", &morphosml::set_seed, py::arg("seed"), "Set global seed for reproducible computations");
    m.def("get_seed", &morphosml::get_seed, "Get current global seed");

    // Vector bindings
    py::class_<morphosml::Vector>(m, "Vector")
        .def(py::init<>())
        .def(py::init<size_t>())
        .def(py::init<std::vector<double>>())
        .def("size", &morphosml::Vector::size)
        .def("__len__", &morphosml::Vector::size)
        .def("__getitem__", [](morphosml::Vector& v, size_t i) { return v.at(i); })
        .def("__setitem__", [](morphosml::Vector& v, size_t i, double val) { v.at(i) = val; })
        .def("dot", &morphosml::Vector::dot)
        .def("norm", &morphosml::Vector::norm)
        .def("normalized", &morphosml::Vector::normalized)
        .def("__add__", [](const morphosml::Vector& a, const morphosml::Vector& b) {
            return a + b;
        })
        .def("__sub__", [](const morphosml::Vector& a, const morphosml::Vector& b) {
            return a - b;
        })
        .def("__mul__", [](const morphosml::Vector& a, double scalar) {
            return a * scalar;
        })
        .def("__rmul__", [](const morphosml::Vector& a, double scalar) {
            return a * scalar;
        })
        .def("to_list", &morphosml::Vector::to_std);

    // HPC Matrix bindings with Buffer Protocol support (zero-copy view to NumPy)
    py::class_<morphosml::Matrix>(m, "Matrix", py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<size_t, size_t>(), py::arg("rows"), py::arg("cols"))
        .def(py::init<std::vector<std::vector<double>>>())
        .def("rows", &morphosml::Matrix::rows)
        .def("cols", &morphosml::Matrix::cols)
        .def("size", &morphosml::Matrix::size)
        .def("__call__", [](morphosml::Matrix& mat, size_t i, size_t j) -> double& {
            return mat(i, j);
        })
        .def("__getitem__", [](morphosml::Matrix& mat, py::tuple idx) -> double {
            if (idx.size() != 2) throw std::invalid_argument("Expected 2D index (i, j)");
            return mat(idx[0].cast<size_t>(), idx[1].cast<size_t>());
        })
        .def("__setitem__", [](morphosml::Matrix& mat, py::tuple idx, double val) {
            if (idx.size() != 2) throw std::invalid_argument("Expected 2D index (i, j)");
            mat(idx[0].cast<size_t>(), idx[1].cast<size_t>()) = val;
        })
        .def("transpose", &morphosml::Matrix::transpose)
        .def("__mul__", [](const morphosml::Matrix& a, const morphosml::Matrix& b) {
            return a * b;
        })
        .def("__mul__", [](const morphosml::Matrix& a, double scalar) {
            return a * scalar;
        })
        .def("__add__", [](const morphosml::Matrix& a, const morphosml::Matrix& b) {
            return a + b;
        })
        .def("__sub__", [](const morphosml::Matrix& a, const morphosml::Matrix& b) {
            return a - b;
        })
        .def_static("identity", &morphosml::Matrix::identity)
        .def_static("zeros", &morphosml::Matrix::zeros)
        .def_static("random", py::overload_cast<size_t, size_t>(&morphosml::Matrix::random))
        .def_static("random", py::overload_cast<size_t, size_t, uint64_t>(&morphosml::Matrix::random),
                    py::arg("rows"), py::arg("cols"), py::arg("seed"))
        .def("to_list", &morphosml::Matrix::to_std)
        .def_buffer([](morphosml::Matrix &mat) -> py::buffer_info {
            return py::buffer_info(
                mat.data(),
                sizeof(double),
                py::format_descriptor<double>::format(),
                2,
                { mat.rows(), mat.cols() },
                { sizeof(double) * mat.cols(), sizeof(double) }
            );
        });

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

    // LinearRegression bindings (Issue #9)
    py::class_<morphosml::LinearRegression>(m, "LinearRegression")
        .def(py::init<double, size_t, bool>(),
             py::arg("learning_rate") = 0.01,
             py::arg("epochs") = 1000,
             py::arg("fit_intercept") = true)
        .def("fit", [](morphosml::LinearRegression& lr, py::array_t<double> X, std::vector<double> y) {
            lr.fit(numpy_to_matrix(X), y);
        })
        .def("predict", [](const morphosml::LinearRegression& lr, py::array_t<double> X) {
            return lr.predict(numpy_to_matrix(X));
        })
        .def("score", [](const morphosml::LinearRegression& lr, py::array_t<double> X, std::vector<double> y) {
            return lr.score(numpy_to_matrix(X), y);
        })
        .def("compute_cost", [](const morphosml::LinearRegression& lr, py::array_t<double> X, std::vector<double> y) {
            return lr.compute_cost(numpy_to_matrix(X), y);
        })
        .def_property_readonly("weights", &morphosml::LinearRegression::weights)
        .def_property_readonly("bias", &morphosml::LinearRegression::bias)
        .def_property_readonly("learning_rate", &morphosml::LinearRegression::learning_rate)
        .def_property_readonly("epochs", &morphosml::LinearRegression::epochs)
        .def_property_readonly("is_fitted", &morphosml::LinearRegression::is_fitted);

    // LogisticRegression bindings (Issue #9)
    py::class_<morphosml::LogisticRegression>(m, "LogisticRegression")
        .def(py::init<double, size_t, bool>(),
             py::arg("learning_rate") = 0.01,
             py::arg("epochs") = 1000,
             py::arg("fit_intercept") = true)
        .def("fit", [](morphosml::LogisticRegression& lr, py::array_t<double> X, std::vector<int> y) {
            lr.fit(numpy_to_matrix(X), y);
        })
        .def("predict", [](const morphosml::LogisticRegression& lr, py::array_t<double> X, double threshold) {
            return lr.predict(numpy_to_matrix(X), threshold);
        }, py::arg("X"), py::arg("threshold") = 0.5)
        .def("predict_proba", [](const morphosml::LogisticRegression& lr, py::array_t<double> X) {
            return lr.predict_proba(numpy_to_matrix(X));
        })
        .def("score", [](const morphosml::LogisticRegression& lr, py::array_t<double> X, std::vector<int> y) {
            return lr.score(numpy_to_matrix(X), y);
        })
        .def("compute_loss", [](const morphosml::LogisticRegression& lr, py::array_t<double> X, std::vector<int> y) {
            return lr.compute_loss(numpy_to_matrix(X), y);
        })
        .def_property_readonly("weights", &morphosml::LogisticRegression::weights)
        .def_property_readonly("bias", &morphosml::LogisticRegression::bias)
        .def_property_readonly("learning_rate", &morphosml::LogisticRegression::learning_rate)
        .def_property_readonly("epochs", &morphosml::LogisticRegression::epochs)
        .def_property_readonly("is_fitted", &morphosml::LogisticRegression::is_fitted);

    // Metrics bindings
    m.def("accuracy", &morphosml::accuracy, "Calculate classification accuracy");
    m.def("mse", py::overload_cast<const std::vector<double>&, const std::vector<double>&>(&morphosml::mse),
          "Calculate mean squared error (regression)");
    m.def("r2_score", &morphosml::r2_score, "Calculate R-squared score (regression)");

    // HPC TensorView (non-owning contiguous 2D view with zero-copy buffer protocol)
    py::class_<morphosml::TensorView>(m, "TensorView", py::buffer_protocol())
        .def(py::init<>())
        .def("rows", &morphosml::TensorView::rows)
        .def("cols", &morphosml::TensorView::cols)
        .def("stride", &morphosml::TensorView::stride)
        .def("size", &morphosml::TensorView::size)
        .def("empty", &morphosml::TensorView::empty)
        .def("to_matrix", &morphosml::TensorView::to_matrix)
        .def("__call__", [](const morphosml::TensorView& tv, size_t r, size_t c) {
            return tv(r, c);
        })
        .def("__getitem__", [](const morphosml::TensorView& tv, py::tuple idx) -> double {
            if (idx.size() != 2) throw std::invalid_argument("Expected 2D index (i, j)");
            return tv(idx[0].cast<size_t>(), idx[1].cast<size_t>());
        })
        .def_buffer([](morphosml::TensorView& tv) -> py::buffer_info {
            return py::buffer_info(
                const_cast<double*>(tv.data()),
                sizeof(double),
                py::format_descriptor<double>::format(),
                2,
                { tv.rows(), tv.cols() },
                { sizeof(double) * tv.stride(), sizeof(double) },
                true
            );
        });

    m.def("tensor_view_from_numpy", [](py::array_t<double> arr) {
        py::buffer_info buf = arr.request();
        if (buf.ndim != 2) throw std::invalid_argument("The input NumPy array must be 2-dimensional.");
        size_t rows = static_cast<size_t>(buf.shape[0]);
        size_t cols = static_cast<size_t>(buf.shape[1]);
        size_t stride = static_cast<size_t>(buf.strides[0] / sizeof(double));
        return morphosml::TensorView(
            static_cast<const double*>(buf.ptr),
            rows,
            cols,
            stride
        );
    }, "Create a zero-copy TensorView wrapping a 2D NumPy array buffer");

    // IngestionCursor bindings
    py::class_<morphosml::data::IngestionCursor>(m, "IngestionCursor")
        .def(py::init<>())
        .def_readwrite("epoch", &morphosml::data::IngestionCursor::epoch)
        .def_readwrite("sample_offset", &morphosml::data::IngestionCursor::sample_offset)
        .def_readwrite("checksum", &morphosml::data::IngestionCursor::checksum)
        .def("to_string", &morphosml::data::IngestionCursor::to_string)
        .def_static("from_string", &morphosml::data::IngestionCursor::from_string)
        .def("__repr__", [](const morphosml::data::IngestionCursor& c) {
            return "<IngestionCursor epoch=" + std::to_string(c.epoch) +
                   " sample_offset=" + std::to_string(c.sample_offset) +
                   " checksum=" + std::to_string(c.checksum) + ">";
        });

    // IdempotentSampler bindings
    py::class_<morphosml::data::IdempotentSampler>(m, "IdempotentSampler")
        .def(py::init<uint64_t>(), py::arg("seed") = 42)
        .def("generate_indices", &morphosml::data::IdempotentSampler::generate_indices,
             py::arg("total_samples"), py::arg("epoch") = 0)
        .def_property("seed", &morphosml::data::IdempotentSampler::seed, &morphosml::data::IdempotentSampler::set_seed);

    // MMapBuffer bindings
    py::class_<morphosml::data::MMapBuffer>(m, "MMapBuffer")
        .def(py::init<const std::string&>(), py::arg("filepath"))
        .def("rows", &morphosml::data::MMapBuffer::rows)
        .def("cols", &morphosml::data::MMapBuffer::cols)
        .def("size", &morphosml::data::MMapBuffer::size)
        .def("checksum", &morphosml::data::MMapBuffer::checksum)
        .def("filepath", &morphosml::data::MMapBuffer::filepath)
        .def("get_slice", &morphosml::data::MMapBuffer::get_slice, py::arg("start_row"), py::arg("num_rows"))
        .def("full_view", &morphosml::data::MMapBuffer::full_view)
        .def_static("write_mldat", &morphosml::data::MMapBuffer::write_mldat, py::arg("filepath"), py::arg("mat"))
        .def_static("write_mldat_numpy", [](const std::string& filepath, py::array_t<double> arr) {
            morphosml::Matrix mat = numpy_to_matrix(arr);
            morphosml::data::MMapBuffer::write_mldat(filepath, mat);
        }, py::arg("filepath"), py::arg("arr"));
}
