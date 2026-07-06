// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_rotate_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ttnn/operations/experimental/fused_rotate/fused_rotate.hpp"

namespace ttnn::operations::experimental::fr_detail {

void bind_fused_rotate(nb::module_& mod) {
    mod.def(
        "fused_rotate",
        [](const ttnn::Tensor& x_flat,
           const ttnn::Tensor& coef_exp,
           uint32_t n_in,
           uint32_t n_out,
           uint32_t W,
           const std::vector<uint32_t>& deg,
           const std::vector<uint32_t>& ks,
           const std::vector<uint32_t>& js) {
            return ttnn::operations::experimental::fused_rotate(x_flat, coef_exp, n_in, n_out, W, deg, ks, js);
        },
        nb::arg("x_flat").noconvert(),
        nb::arg("coef_exp").noconvert(),
        nb::arg("n_in"),
        nb::arg("n_out"),
        nb::arg("W"),
        nb::arg("deg"),
        nb::arg("ks"),
        nb::arg("js"),
        "Fused per-edge sparse Wigner rotation (all nnz MACs in one kernel launch).");
}

}  // namespace ttnn::operations::experimental::fr_detail
