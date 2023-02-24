// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Cartesian - Euclidean vector space as Lie group

#pragma once
#include "sophus/common/types.h"

namespace sophus {
namespace lie {

template <class TScalar>
class ComplexNumberImpl {
 public:
  using Scalar = TScalar;

  static auto multiplication(
      Eigen::Vector<Scalar, 2> const& lhs_real_imag,
      Eigen::Vector<Scalar, 2> const& rhs_real_imag)
      -> Eigen::Vector<Scalar, 2> {
    // complex multiplication
    return Eigen::Vector<Scalar, 2>(
        lhs_real_imag.x() * rhs_real_imag.x() -
            lhs_real_imag.y() * rhs_real_imag.y(),
        lhs_real_imag.x() * rhs_real_imag.y() +
            lhs_real_imag.y() * rhs_real_imag.x());
  }

  static auto norm(Eigen::Vector<Scalar, 2> const& real_imag)
      -> Eigen::Vector<Scalar, 2> {
    using std::hypot;
    return hypot(real_imag.x(), real_imag.y());
  }
};

}  // namespace lie
}  // namespace sophus
