#include <torch/extension.h>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <sstream>
#include <vector>

namespace py = pybind11;

// The mcsi function
torch::Tensor mcsi(
    const torch::Tensor& input,
    py::object dim,
    bool keepdim
) {
    torch::Tensor mean, n_plus, n_minus, n_float, mcsi_value;
    std::vector<int64_t> dims;

    if (dim.is_none()) {
        dims = {};
    }
    else if (py::isinstance<py::int_>(dim)) {
        dims.push_back(dim.cast<int64_t>());
    }
    else if (py::isinstance<py::iterable>(dim)) {
        dims = dim.cast<std::vector<int64_t>>();
    }
    else {
        TORCH_CHECK(false, "Invalid type for 'dim'. Expected int, list or tuple.");
    }

    if (!dims.empty()) {
        for (auto& d : dims) {
            if (d < 0) {
                d += input.dim();
            }
        }
        size_t original_size = dims.size();
        std::sort(dims.begin(), dims.end());
        dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
        if (dims.size() < original_size) {
            std::stringstream ss;
            ss << "Duplicate dimensions detected and removed. Final dimensions: [";
            for (size_t i = 0; i < dims.size(); ++i) {
                ss << dims[i];
                if (i < dims.size() - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
            TORCH_WARN(ss.str());
        }

        mean = input.mean(dims, /*keepdim=*/true);
        auto greater_than_mean = input > mean;
        auto less_than_mean = input < mean;
        n_plus = greater_than_mean.sum(dims, keepdim);
        n_minus = less_than_mean.sum(dims, keepdim);
        int64_t n = 1;
        for (int64_t d : dims) {
            n *= input.size(d);
        }
        n_float = torch::full_like(n_plus, n, input.options().dtype(at::kFloat));
        mcsi_value = (n_minus - n_plus).toType(at::kFloat) / n_float;

        if (keepdim) {
            std::vector<int64_t> new_shape(input.sizes().vec());
            for (int64_t d : dims) {
                new_shape[d] = 1;
            }
            mcsi_value = mcsi_value.reshape(torch::IntArrayRef(new_shape));
        }

    } else {
        mean = input.mean();
        auto greater_than_mean = input > mean;
        auto less_than_mean = input < mean;
        n_plus = greater_than_mean.sum();
        n_minus = less_than_mean.sum();
        n_float = torch::tensor(input.numel(), input.options().dtype(at::kFloat));
        mcsi_value = (n_minus - n_plus).toType(at::kFloat) / n_float;

        if (keepdim) {
            std::vector<int64_t> new_shape(input.dim(), 1);
            mcsi_value = mcsi_value.reshape(torch::IntArrayRef(new_shape));
        }
    }

    return mcsi_value;
}

// Bind the mcsi function to Python
PYBIND11_MODULE(mcsi, m) {
    m.def("mcsi", &mcsi, "Mean-Centered Skewness Index (MCSI) function",
        py::arg("input"), 
        py::arg("dim") = py::none(),
        py::arg("keepdim") = false);
}